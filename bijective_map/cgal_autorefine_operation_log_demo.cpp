#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/number_utils.h>

#include "batch_operation_log_reader.hpp"
#include "vtu_utils.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
using json = nlohmann::json;

template <typename Matrix>
Matrix json_to_matrix(const json& js)
{
    const int rows = js["rows"].get<int>();
    const auto& values = js["values"];
    if (rows != static_cast<int>(values.size())) {
        throw std::runtime_error("JSON matrix row count mismatch.");
    }
    const int cols = values[0].size();

    Matrix mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (static_cast<int>(values[i].size()) != cols) {
            throw std::runtime_error("JSON matrix column count mismatch.");
        }
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = values[i][j];
        }
    }
    return mat;
}

namespace {
struct TriangleTrackingVisitor : PMP::Autorefinement::Default_visitor
{
    TriangleTrackingVisitor() = default;
    explicit TriangleTrackingVisitor(std::vector<std::size_t>& mapping)
        : m_mapping(&mapping)
    {}

    void number_of_output_triangles(std::size_t nbt)
    {
        if (m_mapping == nullptr) {
            return;
        }
        m_mapping->assign(nbt, static_cast<std::size_t>(-1));
    }

    void verbatim_triangle_copy(std::size_t tgt_id, std::size_t src_id)
    {
        store_mapping(tgt_id, src_id);
    }

    void new_subtriangle(std::size_t tgt_id, std::size_t src_id) { store_mapping(tgt_id, src_id); }

private:
    void store_mapping(std::size_t tgt_id, std::size_t src_id)
    {
        if (m_mapping == nullptr) {
            return;
        }
        if (tgt_id >= m_mapping->size()) {
            m_mapping->resize(tgt_id + 1, static_cast<std::size_t>(-1));
        }
        (*m_mapping)[tgt_id] = src_id;
    }

    std::vector<std::size_t>* m_mapping = nullptr;
};

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Triangle = std::array<std::size_t, 3>;

struct TetTriangle
{
    Triangle triangle;
    std::vector<Eigen::Index> tet_indices;
};

struct SampledVertex
{
    std::size_t point_index;
    Eigen::Vector4d barycentric;
    Eigen::Vector3d position;
    Eigen::Index tet_index;
};

Eigen::MatrixXd json_to_vertex_matrix(const json& node)
{
    return json_to_matrix<Eigen::MatrixXd>(node);
}

Eigen::MatrixXi json_to_tet_matrix(const json& node)
{
    return json_to_matrix<Eigen::MatrixXi>(node);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> load_first_operation_mesh(
    const std::filesystem::path& logs_dir)
{
    BatchOperationLogReader reader(logs_dir);
    const auto total_ops = reader.get_total_operations();
    if (total_ops == 0) {
        throw std::runtime_error("No operations found in directory: " + logs_dir.string());
    }

    const json operation = reader.get_operation(0);
    if (operation.empty()) {
        throw std::runtime_error("Failed to read first operation from: " + logs_dir.string());
    }

    if (!operation.contains("T_before") || !operation.contains("V_before")) {
        throw std::runtime_error("Operation log missing T_before or V_before entries.");
    }

    Eigen::MatrixXi T_before = json_to_tet_matrix(operation["T_before"]);
    Eigen::MatrixXd V_before = json_to_vertex_matrix(operation["V_before"]);
    if (T_before.cols() != 4 || V_before.cols() != 3) {
        throw std::runtime_error("Unexpected dimensions for T_before/V_before.");
    }
    return {V_before, T_before};
}

std::vector<Point> eigen_vertices_to_points(const Eigen::MatrixXd& V)
{
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(V.rows()));
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        points.emplace_back(V(i, 0), V(i, 1), V(i, 2));
    }
    return points;
}

std::vector<TetTriangle> extract_all_tet_triangles(const Eigen::MatrixXi& T)
{
    struct FaceEntry
    {
        Triangle oriented = {0, 0, 0};
        std::vector<Eigen::Index> tet_indices;
        bool has_orientation = false;
    };

    const int local_faces[4][3] = {
        {1, 2, 3}, // face opposite vertex 0
        {0, 3, 2}, // face opposite vertex 1
        {0, 1, 3}, // face opposite vertex 2
        {0, 2, 1} // face opposite vertex 3
    };

    std::map<std::array<int, 3>, FaceEntry> face_map;

    for (Eigen::Index tet = 0; tet < T.rows(); ++tet) {
        const auto v0 = static_cast<int>(T(tet, 0));
        const auto v1 = static_cast<int>(T(tet, 1));
        const auto v2 = static_cast<int>(T(tet, 2));
        const auto v3 = static_cast<int>(T(tet, 3));
        const int tet_vertices[4] = {v0, v1, v2, v3};

        for (const auto& face : local_faces) {
            Triangle oriented = {
                static_cast<std::size_t>(tet_vertices[face[0]]),
                static_cast<std::size_t>(tet_vertices[face[1]]),
                static_cast<std::size_t>(tet_vertices[face[2]])};

            std::array<int, 3> key = {
                tet_vertices[face[0]],
                tet_vertices[face[1]],
                tet_vertices[face[2]]};
            std::sort(key.begin(), key.end());

            auto& entry = face_map[key];
            if (!entry.has_orientation) {
                entry.oriented = oriented;
                entry.has_orientation = true;
            }
            entry.tet_indices.push_back(tet);
        }
    }

    std::vector<TetTriangle> tet_faces;
    tet_faces.reserve(face_map.size());
    for (auto& [key, entry] : face_map) {
        tet_faces.push_back(TetTriangle{entry.oriented, entry.tet_indices});
    }
    return tet_faces;
}

Eigen::Vector4d random_barycentric(std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Eigen::Vector4d weights;
    do {
        for (int i = 0; i < 4; ++i) {
            weights[i] = dist(rng);
        }
    } while (weights.sum() == 0.0);

    weights /= weights.sum();
    // Bias towards interior by lifting weights slightly
    constexpr double min_weight = 0.05;
    for (int i = 0; i < 4; ++i) {
        weights[i] = std::max(weights[i], min_weight);
    }
    weights /= weights.sum();
    return weights;
}

SampledVertex sample_point_in_tet(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T,
    Eigen::Index tet_index,
    std::mt19937& rng,
    std::vector<Point>& points)
{
    Eigen::Vector4d bc = random_barycentric(rng);
    Eigen::Matrix<double, 4, 3> tet_vertices;
    for (int i = 0; i < 4; ++i) {
        tet_vertices.row(i) = V.row(T(tet_index, i));
    }

    Eigen::Vector3d p = bc[0] * tet_vertices.row(0) + bc[1] * tet_vertices.row(1) +
                        bc[2] * tet_vertices.row(2) + bc[3] * tet_vertices.row(3);

    points.emplace_back(p.x(), p.y(), p.z());
    SampledVertex result;
    result.point_index = points.size() - 1;
    result.barycentric = bc;
    result.position = p;
    result.tet_index = tet_index;
    return result;
}

Triangle build_sampled_triangle(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T,
    std::vector<Point>& points,
    std::vector<SampledVertex>& sampled_vertices)
{
    if (T.rows() < 3) {
        throw std::runtime_error("Need at least three tetrahedra to sample triangle vertices.");
    }

    std::mt19937 rng(1337);
    std::uniform_int_distribution<Eigen::Index> tet_dist(0, T.rows() - 1);
    std::set<Eigen::Index> chosen;
    while (chosen.size() < 3) {
        chosen.insert(tet_dist(rng));
    }

    Triangle tri{};
    int idx = 0;
    for (Eigen::Index tet_id : chosen) {
        SampledVertex vertex = sample_point_in_tet(V, T, tet_id, rng, points);
        tri[idx] = vertex.point_index;
        sampled_vertices.push_back(vertex);
        ++idx;
    }
    return tri;
}

Eigen::MatrixXd to_vertex_matrix(const std::vector<Point>& pts)
{
    Eigen::MatrixXd V(pts.size(), 3);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(pts.size()); ++i) {
        V(i, 0) = CGAL::to_double(pts[i].x());
        V(i, 1) = CGAL::to_double(pts[i].y());
        V(i, 2) = CGAL::to_double(pts[i].z());
    }
    return V;
}

Eigen::MatrixXi to_face_matrix(const std::vector<Triangle>& tris)
{
    Eigen::MatrixXi F(tris.size(), 3);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(tris.size()); ++i) {
        F(i, 0) = static_cast<int>(tris[i][0]);
        F(i, 1) = static_cast<int>(tris[i][1]);
        F(i, 2) = static_cast<int>(tris[i][2]);
    }
    return F;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::filesystem::path logs_dir =
            (argc > 1) ? std::filesystem::path(argv[1])
                       : std::filesystem::path("../build/operation_log_sphere_coarse");

        std::cout << "Reading operation logs from: " << logs_dir << '\n';
        auto [V_before, T_before] = load_first_operation_mesh(logs_dir);
        std::cout << "Loaded tet mesh: " << V_before.rows() << " vertices, " << T_before.rows()
                  << " tetrahedra\n";

        std::vector<Point> points = eigen_vertices_to_points(V_before);
        std::vector<SampledVertex> sampled_vertices;
        Triangle sampled_triangle =
            build_sampled_triangle(V_before, T_before, points, sampled_vertices);

        auto tet_triangles = extract_all_tet_triangles(T_before);
        std::cout << "Collected " << tet_triangles.size()
                  << " tetrahedral faces as triangle soup input\n";

        std::vector<Triangle> triangles;
        triangles.reserve(tet_triangles.size() + 1);
        std::vector<std::vector<int>> triangle_parent_tets;
        triangle_parent_tets.reserve(tet_triangles.size() + 1);
        for (const TetTriangle& face : tet_triangles) {
            triangles.push_back(face.triangle);
            std::vector<int> parents;
            parents.reserve(face.tet_indices.size());
            for (Eigen::Index tet_id : face.tet_indices) {
                parents.push_back(static_cast<int>(tet_id));
            }
            triangle_parent_tets.push_back(std::move(parents));
        }
        const std::size_t inserted_triangle_index = triangles.size();
        triangles.push_back(sampled_triangle);
        triangle_parent_tets.push_back(std::vector<int>{-1}); // sampled test triangle

        const std::vector<Point> original_points = points;
        const std::vector<Triangle> original_triangles = triangles;
        const std::vector<std::vector<int>> original_triangle_parent_tets = triangle_parent_tets;

        const bool had_initial_intersections =
            PMP::does_triangle_soup_self_intersect(points, triangles);

        std::vector<std::vector<std::size_t>> working_triangles;
        working_triangles.reserve(triangles.size());
        for (const Triangle& tri : triangles) {
            working_triangles.push_back({tri[0], tri[1], tri[2]});
        }

        std::vector<std::size_t> triangle_source_ids;
        TriangleTrackingVisitor visitor(triangle_source_ids);
        PMP::autorefine_triangle_soup(
            points,
            working_triangles,
            CGAL::parameters::visitor(visitor).apply_iterative_snap_rounding(true));

        triangles.clear();
        triangles.reserve(working_triangles.size());
        std::vector<std::size_t> filtered_source_ids;
        filtered_source_ids.reserve(working_triangles.size());
        for (std::size_t i = 0; i < working_triangles.size(); ++i) {
            const auto& tri = working_triangles[i];
            if (tri.size() != 3) {
                std::cerr << "Warning: encountered triangle with " << tri.size()
                          << " vertices after autorefinement; skipping.\n";
                continue;
            }
            triangles.push_back(Triangle{tri[0], tri[1], tri[2]});
            if (i < triangle_source_ids.size()) {
                filtered_source_ids.push_back(triangle_source_ids[i]);
            }
        }
        triangle_source_ids.swap(filtered_source_ids);

        const std::size_t invalid_id = static_cast<std::size_t>(-1);

        std::vector<std::vector<int>> refined_triangle_parent_tets(triangles.size());
        for (std::size_t i = 0; i < triangles.size() && i < triangle_source_ids.size(); ++i) {
            const std::size_t src_id = triangle_source_ids[i];
            if (src_id == invalid_id || src_id >= triangle_parent_tets.size()) {
                continue;
            }
            refined_triangle_parent_tets[i] = triangle_parent_tets[src_id];
        }

        std::vector<std::vector<std::size_t>> vertex_incident_triangles(points.size());
        for (std::size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
            const Triangle& tri = triangles[tri_idx];
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const std::size_t v_id = tri[corner];
                if (v_id >= vertex_incident_triangles.size()) {
                    continue;
                }
                vertex_incident_triangles[v_id].push_back(tri_idx);
            }
        }

        std::vector<std::set<int>> vertex_tet_sets(points.size());
        for (const SampledVertex& vertex : sampled_vertices) {
            if (vertex.point_index < vertex_tet_sets.size()) {
                vertex_tet_sets[vertex.point_index].insert(static_cast<int>(vertex.tet_index));
            }
        }
        for (std::size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
            const Triangle& tri = triangles[tri_idx];
            if (tri_idx >= refined_triangle_parent_tets.size()) {
                continue;
            }
            const auto& parents = refined_triangle_parent_tets[tri_idx];
            for (int tet_id : parents) {
                if (tet_id < 0) {
                    continue;
                }
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t v_id = tri[corner];
                    if (v_id < vertex_tet_sets.size()) {
                        vertex_tet_sets[v_id].insert(tet_id);
                    }
                }
            }
        }

        const bool intersection_free = !PMP::does_triangle_soup_self_intersect(points, triangles);

        std::cout << "Initial soup had intersections: "
                  << (had_initial_intersections ? "yes" : "no") << '\n';
        std::cout << "After autorefinement: "
                  << (intersection_free ? "no remaining intersections." : "still intersects.")
                  << '\n';
        std::cout << "Output point count: " << points.size() << '\n';
        std::cout << "Output triangle count: " << triangles.size() << '\n';

        Eigen::VectorXi triangle_origin_ids = Eigen::VectorXi::Constant(triangles.size(), -1);
        for (Eigen::Index i = 0; i < triangle_origin_ids.size() &&
                                 i < static_cast<Eigen::Index>(triangle_source_ids.size());
             ++i) {
            const std::size_t src_id = triangle_source_ids[i];
            if (src_id == invalid_id) {
                continue;
            }
            if (src_id > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                std::cerr << "Warning: source triangle id " << src_id
                          << " exceeds Int32 range; writing as -1 in VTU.\n";
                continue;
            }
            triangle_origin_ids(static_cast<int>(i)) = static_cast<int>(src_id);
        }

        Eigen::VectorXi triangle_origin_tet = Eigen::VectorXi::Constant(triangles.size(), -1);
        for (Eigen::Index i = 0; i < triangle_origin_tet.size(); ++i) {
            if (i < static_cast<Eigen::Index>(refined_triangle_parent_tets.size())) {
                const auto& parents = refined_triangle_parent_tets[static_cast<std::size_t>(i)];
                triangle_origin_tet(i) = parents.empty() ? -1 : parents.front();
            }
        }

        const Eigen::MatrixXd V_original = to_vertex_matrix(original_points);
        const Eigen::MatrixXi F_original = to_face_matrix(original_triangles);
        const Eigen::MatrixXd V_refined = to_vertex_matrix(points);
        const Eigen::MatrixXi F_refined = to_face_matrix(triangles);

        Eigen::VectorXi original_triangle_parent_vec =
            Eigen::VectorXi::Constant(original_triangles.size(), -1);
        for (Eigen::Index i = 0; i < original_triangle_parent_vec.size(); ++i) {
            if (i < static_cast<Eigen::Index>(original_triangle_parent_tets.size())) {
                const auto& parents =
                    original_triangle_parent_tets[static_cast<std::size_t>(i)];
                original_triangle_parent_vec(i) = parents.empty() ? -1 : parents.front();
            }
        }

        const std::string before_path = "operation_log_autorefine_before.vtu";
        const std::string after_path = "operation_log_autorefine_after.vtu";
        const std::string before_tet_path = "operation_log_autorefine_before_tet.vtu";
        const std::string after_tet_path = "operation_log_autorefine_after_tet.vtu";

        vtu_utils::write_triangle_mesh_to_vtu(V_original, F_original, before_path);
        vtu_utils::write_triangle_mesh_to_vtu(
            V_refined,
            F_refined,
            after_path,
            triangle_origin_ids.size() == F_refined.rows() ? &triangle_origin_ids : nullptr,
            "origin_triangle_id");
        vtu_utils::write_triangle_mesh_to_vtu(
            V_original,
            F_original,
            before_tet_path,
            original_triangle_parent_vec.size() == F_original.rows()
                ? &original_triangle_parent_vec
                : nullptr,
            "origin_tet_id");
        vtu_utils::write_triangle_mesh_to_vtu(
            V_refined,
            F_refined,
            after_tet_path,
            triangle_origin_tet.size() == F_refined.rows() ? &triangle_origin_tet : nullptr,
            "origin_tet_id");

        std::cout << "Wrote VTU snapshots:\n"
                  << "  initial soup -> " << before_path << '\n'
                  << "  initial soup (origin_tet_id) -> " << before_tet_path << '\n'
                  << "  refined soup -> " << after_path << '\n'
                  << "  refined soup (origin_tet_id) -> " << after_tet_path << '\n';

        std::vector<Triangle> refined_sampled_triangles;
        std::vector<std::size_t> refined_sampled_indices;
        for (std::size_t i = 0; i < triangles.size() && i < triangle_source_ids.size(); ++i) {
            if (triangle_source_ids[i] == inserted_triangle_index) {
                refined_sampled_triangles.push_back(triangles[i]);
                refined_sampled_indices.push_back(i);
            }
        }

        if (!refined_sampled_triangles.empty()) {
            const Eigen::MatrixXi F_subset = to_face_matrix(refined_sampled_triangles);
            const std::string subset_path = "operation_log_autorefine_sampled_triangle.vtu";
            Eigen::VectorXi subset_tet_ids =
                Eigen::VectorXi::Constant(F_subset.rows(), -1);
            std::cout << "Refined sampled triangle written to: " << subset_path << '\n';

            std::set<std::size_t> sampled_vertex_ids;
            for (std::size_t local_idx = 0; local_idx < refined_sampled_indices.size(); ++local_idx) {
                const std::size_t tri_idx = refined_sampled_indices[local_idx];
                const Triangle& tri = triangles[tri_idx];
                std::cout << "Sample triangle piece " << local_idx
                          << " corresponds to refined triangle " << tri_idx << " [vertices "
                          << tri[0] << ", " << tri[1] << ", " << tri[2] << "]\n";

                std::set<int> common_tets;
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t v_id = tri[corner];
                    if (v_id >= vertex_tet_sets.size()) {
                        common_tets.clear();
                        break;
                    }
                    const auto& tet_set = vertex_tet_sets[v_id];
                    if (tet_set.empty()) {
                        common_tets.clear();
                        break;
                    }
                    if (corner == 0) {
                        common_tets = tet_set;
                    } else {
                        std::set<int> temp;
                        std::set_intersection(
                            common_tets.begin(),
                            common_tets.end(),
                            tet_set.begin(),
                            tet_set.end(),
                            std::inserter(temp, temp.begin()));
                        common_tets.swap(temp);
                    }
                    if (common_tets.empty()) {
                        break;
                    }
                }

                const int assigned_tet =
                    common_tets.empty() ? -1 : *common_tets.begin();
                if (local_idx < static_cast<std::size_t>(subset_tet_ids.size())) {
                    subset_tet_ids(static_cast<int>(local_idx)) = assigned_tet;
                }
                if (tri_idx < static_cast<std::size_t>(triangle_origin_tet.size())) {
                    triangle_origin_tet(static_cast<int>(tri_idx)) = assigned_tet;
                }

                std::set<int> piece_partner_tets;
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t v_id = tri[corner];
                    sampled_vertex_ids.insert(v_id);
                    std::set<int> vertex_partner_tets;
                    if (v_id < vertex_tet_sets.size()) {
                        vertex_partner_tets.insert(
                            vertex_tet_sets[v_id].begin(),
                            vertex_tet_sets[v_id].end());
                        piece_partner_tets.insert(
                            vertex_tet_sets[v_id].begin(),
                            vertex_tet_sets[v_id].end());
                    }

                    const Point& p = points[v_id];
                    std::cout << "    vertex " << v_id << " (" << CGAL::to_double(p.x()) << ", "
                              << CGAL::to_double(p.y()) << ", " << CGAL::to_double(p.z())
                              << ") shared with tets: ";
                    if (vertex_partner_tets.empty()) {
                        std::cout << "none";
                    } else {
                        bool first = true;
                        for (int tet_id : vertex_partner_tets) {
                            if (!first) {
                                std::cout << ", ";
                            }
                            std::cout << tet_id;
                            first = false;
                        }
                    }
                    std::cout << '\n';
                }

                std::cout << "    assigned tet id (intersection): ";
                if (assigned_tet == -1) {
                    std::cout << "none";
                } else {
                    std::cout << assigned_tet;
                }
                std::cout << '\n';

                std::cout << "    aggregated partner tets: ";
                if (piece_partner_tets.empty()) {
                    std::cout << "none";
                } else {
                    bool first = true;
                    for (int tet_id : piece_partner_tets) {
                        if (!first) {
                            std::cout << ", ";
                        }
                        std::cout << tet_id;
                        first = false;
                    }
                }
                std::cout << '\n';
            }

            vtu_utils::write_triangle_mesh_to_vtu(
                V_refined,
                F_subset,
                subset_path,
                subset_tet_ids.size() == F_subset.rows() ? &subset_tet_ids : nullptr,
                "sampled_tet_id");

            std::cout << "Unique vertices belonging to test triangle fragments:\n";
            for (std::size_t v_id : sampled_vertex_ids) {
                const Point& p = points[v_id];
                std::cout << "  vertex " << v_id << " -> (" << CGAL::to_double(p.x()) << ", "
                          << CGAL::to_double(p.y()) << ", " << CGAL::to_double(p.z()) << ")\n";
            }
        } else {
            std::cout << "No refined triangles mapped back to the sampled triangle.\n";
        }

        std::cout << "Sampled triangle came from tetrahedra:\n";
        for (const auto& vertex : sampled_vertices) {
            std::cout << "  tet " << vertex.tet_index << " -> point #" << vertex.point_index
                      << " at (" << vertex.position.transpose() << "), barycentric "
                      << vertex.barycentric.transpose() << '\n';
        }

        return intersection_free ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

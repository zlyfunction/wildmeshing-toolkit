#include <CGAL/number_utils.h>

#include "batch_operation_log_reader.hpp"
#include "cgal_autorefine_sampling.hpp"
#include "cgal_autorefine_utils.hpp"
#include "vtu_utils.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
using cgal_autorefine_demo::autorefine_sampled_triangles;
using cgal_autorefine_demo::AutorefineResult;
using cgal_autorefine_demo::build_sampled_triangles;
using cgal_autorefine_demo::Point;
using cgal_autorefine_demo::SampledPointInput;
using cgal_autorefine_demo::Triangle;

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
using cgal_autorefine_demo::AutorefineResult;
using cgal_autorefine_demo::Point;
using cgal_autorefine_demo::SampledPointInput;
using cgal_autorefine_demo::SampledVertex;
using cgal_autorefine_demo::Triangle;

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

// Convert shared sampling results to SampledPointInput format
void convert_sampling_to_input(
    const std::vector<Eigen::Vector4d>& sampled_barycentrics,
    const std::vector<Eigen::Index>& sampled_tet_indices,
    std::vector<SampledPointInput>& sampled_points)
{
    sampled_points.clear();
    sampled_points.reserve(sampled_barycentrics.size());
    for (std::size_t i = 0; i < sampled_barycentrics.size(); ++i) {
        SampledPointInput input;
        input.tet_index = sampled_tet_indices[i];
        input.barycentric = sampled_barycentrics[i];
        sampled_points.push_back(input);
    }
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

        std::vector<Eigen::Vector4d> sampled_barycentrics;
        std::vector<Eigen::Index> sampled_tet_indices;
        const int sample_triangle_count = 2;
        Eigen::MatrixXi sampled_faces = build_sampled_triangles(
            T_before,
            sampled_barycentrics,
            sampled_tet_indices,
            sample_triangle_count);

        std::vector<SampledPointInput> sampled_points;
        convert_sampling_to_input(sampled_barycentrics, sampled_tet_indices, sampled_points);
        std::cout << "Sampling " << sampled_faces.rows() << " test triangles ("
                  << sampled_points.size() << " vertices)\n";
        for (size_t i = 0; i < sampled_points.size(); ++i) {
            const auto& pt = sampled_points[i];
            std::cout << "Sampled Point " << i << ": barycentric = [";
            for (int j = 0; j < 4; ++j) {
                std::cout << pt.barycentric[j];
                if (j < 3) std::cout << ", ";
            }
            std::cout << "], tet_index = " << pt.tet_index << std::endl;
        }
        AutorefineResult result =
            autorefine_sampled_triangles(V_before, T_before, sampled_points, sampled_faces);

        const std::size_t tet_face_count =
            result.original_triangles.size() - static_cast<std::size_t>(sampled_faces.rows());
        std::cout << "Collected " << tet_face_count
                  << " tetrahedral faces as triangle soup input\n";

        std::cout << "Initial soup had intersections: "
                  << (result.initial_soup_had_intersections ? "yes" : "no") << '\n';
        std::cout << "After autorefinement: "
                  << (result.refined_soup_is_intersection_free ? "no remaining intersections."
                                                               : "still intersects.")
                  << '\n';
        std::cout << "Output point count: " << result.refined_points.size() << '\n';
        std::cout << "Output triangle count: " << result.refined_triangles.size() << '\n';

        if (result.origin_triangle_ids.size() != result.refined_triangles.size()) {
            std::cerr << "Warning: visitor returned " << result.origin_triangle_ids.size()
                      << " triangle mappings for " << result.refined_triangles.size()
                      << " output triangles.\n";
        }

        for (Eigen::Index i = 0; i < result.origin_triangle_ids.size(); ++i) {
            const int src_triangle = result.origin_triangle_ids(i);
            std::cout << "Triangle " << i << " originates from input triangle " << src_triangle;
            if (src_triangle >= 0 &&
                src_triangle < static_cast<int>(result.original_triangle_parent_tets.size())) {
                const auto& parents =
                    result.original_triangle_parent_tets[static_cast<std::size_t>(src_triangle)];
                if (!parents.empty() && parents.front() != -1) {
                    std::cout << " (tet " << parents.front() << ")";
                }
            }
            std::cout << '\n';
        }

        const Eigen::VectorXi& triangle_origin_ids = result.origin_triangle_ids;
        const Eigen::VectorXi& triangle_origin_tet = result.origin_tet_ids;

        const Eigen::MatrixXd V_original = to_vertex_matrix(result.original_points);
        const Eigen::MatrixXi F_original = to_face_matrix(result.original_triangles);
        const Eigen::MatrixXd V_refined = to_vertex_matrix(result.refined_points);
        const Eigen::MatrixXi F_refined = to_face_matrix(result.refined_triangles);

        Eigen::VectorXi original_triangle_parent_vec =
            Eigen::VectorXi::Constant(result.original_triangles.size(), -1);
        for (Eigen::Index i = 0; i < original_triangle_parent_vec.size(); ++i) {
            const auto& parents = result.original_triangle_parent_tets[static_cast<std::size_t>(i)];
            if (!parents.empty()) {
                original_triangle_parent_vec(i) = parents.front();
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
            original_triangle_parent_vec.size() == F_original.rows() ? &original_triangle_parent_vec
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

        if (!result.sampled_fragment_triangles.empty()) {
            const Eigen::MatrixXi F_subset = to_face_matrix(result.sampled_fragment_triangles);
            const std::string subset_path = "operation_log_autorefine_sampled_triangle.vtu";
            const Eigen::VectorXi& subset_tet_ids = result.sampled_fragment_tet_ids;
            vtu_utils::write_triangle_mesh_to_vtu(
                V_refined,
                F_subset,
                subset_path,
                subset_tet_ids.size() == F_subset.rows() ? &subset_tet_ids : nullptr,
                "sampled_tet_id");
            std::cout << "Refined sampled triangle written to: " << subset_path << '\n';

            std::set<std::size_t> sampled_vertex_ids;
            for (std::size_t local_idx = 0; local_idx < result.sampled_fragment_triangles.size();
                 ++local_idx) {
                const Triangle& tri = result.sampled_fragment_triangles[local_idx];
                const std::size_t tri_idx = result.sampled_fragment_indices[local_idx];
                const int assigned_tet = result.sampled_fragment_tet_ids(local_idx);

                const int source_sample = result.sampled_fragment_source_ids[local_idx];
                std::cout << "Sample triangle piece " << local_idx << " (from test triangle "
                          << source_sample << ") corresponds to refined triangle " << tri_idx
                          << " [vertices " << tri[0] << ", " << tri[1] << ", " << tri[2] << "]\n";

                std::set<int> piece_partner_tets;
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t v_id = tri[corner];
                    sampled_vertex_ids.insert(v_id);
                    const Point& p = result.refined_points[v_id];
                    std::cout << "    vertex " << v_id << " (" << CGAL::to_double(p.x()) << ", "
                              << CGAL::to_double(p.y()) << ", " << CGAL::to_double(p.z())
                              << ") shared with tets: ";
                    const auto& tet_set = result.vertex_tet_sets[v_id];
                    if (tet_set.empty()) {
                        std::cout << "none";
                    } else {
                        bool first = true;
                        for (int tet_id : tet_set) {
                            if (!first) {
                                std::cout << ", ";
                            }
                            std::cout << tet_id;
                            first = false;
                        }
                        piece_partner_tets.insert(tet_set.begin(), tet_set.end());
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

            std::cout << "Unique vertices belonging to test triangle fragments:\n";
            for (std::size_t v_id : sampled_vertex_ids) {
                const Point& p = result.refined_points[v_id];
                std::cout << "  vertex " << v_id << " -> (" << CGAL::to_double(p.x()) << ", "
                          << CGAL::to_double(p.y()) << ", " << CGAL::to_double(p.z()) << ")\n";
            }
        } else {
            std::cout << "No refined triangles mapped back to the sampled triangle.\n";
        }

        std::cout << "Sampled triangle came from tetrahedra:\n";
        for (const auto& vertex : result.sampled_vertices) {
            std::cout << "  tet " << vertex.tet_index << " -> point #" << vertex.point_index
                      << " at (" << vertex.position.transpose() << "), barycentric "
                      << vertex.barycentric.transpose() << '\n';
        }

        return result.refined_soup_is_intersection_free ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

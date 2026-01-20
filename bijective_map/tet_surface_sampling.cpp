#include "tet_surface_sampling.hpp"
#include <cinolib/io/write_OBJ.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/number_utils.h>
#include <igl/parallel_for.h>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "FindPointTetMesh.hpp"
#include "cgal_autorefine_utils.hpp"
#include "cgal_autorefine_utils_rational.hpp"
#include "tet_track_operations.hpp"

namespace tet_surface_sampling {

// Helper function for debugging
void generateAndSaveMesh(
    FastTrimesh& tm,
    const Labels& labels,
    int label_id,
    const std::string& output_filename)
{
    tm.resetTrianglesInfo();
    uint num_tris = 0;
    std::cout << "labels.surface.front().size(): " << labels.surface.front().size() << std::endl;
    if (label_id == -1) {
        // All triangles
        for (uint t_id = 0; t_id < tm.numTris(); t_id++) {
            tm.setTriInfo(t_id, 1);
            num_tris++;
        }
    } else {
        // Specific label
        for (uint t_id = 0; t_id < tm.numTris(); t_id++) {
            if (labels.surface[t_id][label_id]) {
                tm.setTriInfo(t_id, 1);
                num_tris++;
            }
        }
    }

    // Prepare output data
    std::vector<double> out_coords;
    std::vector<uint> out_tris;
    std::vector<std::bitset<NBIT>> out_labels;

    // Get the final result
    getFinalMeshInOder(tm, labels, num_tris, out_coords, out_tris, out_labels);

    // Write to OBJ file
    cinolib::write_OBJ(output_filename.c_str(), out_coords, out_tris, {});
}

namespace {
struct EdgeKey
{
    int v0;
    int v1;
    bool operator==(const EdgeKey& other) const { return v0 == other.v0 && v1 == other.v1; }
};

struct EdgeKeyHash
{
    size_t operator()(const EdgeKey& e) const
    {
        // Cantor pairing-esque hash for two ints
        return (static_cast<size_t>(e.v0) << 32) ^ static_cast<size_t>(e.v1);
    }
};

inline EdgeKey make_edge_key(int a, int b)
{
    if (a > b) std::swap(a, b);
    return {a, b};
}

struct TriangleTrackingVisitor : CGAL::Polygon_mesh_processing::Autorefinement::Default_visitor
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

struct TetAabb
{
    Eigen::Vector3d min;
    Eigen::Vector3d max;
};

bool build_sampled_points_and_faces(
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    double tolerance,
    bool verbose,
    std::vector<cgal_autorefine_demo::SampledPointInputRational>& sampled_points,
    Eigen::MatrixXi& sampled_faces)
{
    using Vector4r = Eigen::Matrix<wmtk::Rational, 4, 1>;
    sampled_points.clear();
    sampled_faces.resize(0, 3);

    if (V_surface.rows() == 0 || F_surface.rows() == 0) {
        return false;
    }

    sampled_points.reserve(V_surface.rows());
    std::vector<int> surface_vertex_to_sampled(V_surface.rows(), -1);

    int missing_vertices = 0;
    for (int i = 0; i < V_surface.rows(); ++i) {
        Eigen::Vector3d p = V_surface.row(i);
        auto [tet_id, bc_double] = findTetContainingPoint(V_out, T_out, p, tolerance);
        if (tet_id < 0) {
            missing_vertices++;
            if (verbose) {
                std::cerr << "Warning: surface vertex " << i
                          << " not found in any tet (skipping)" << std::endl;
            }
            continue;
        }

        Vector4r bc_rational;
        for (int j = 0; j < 4; ++j) {
            double value = bc_double(j);
            if (std::abs(value) < tolerance) {
                value = 0.0;
            } else if (std::abs(1.0 - value) < tolerance) {
                value = 1.0;
            }
            value = std::max(0.0, std::min(1.0, value));
            bc_rational(j) = wmtk::Rational(value);
        }
        wmtk::Rational sum = bc_rational.sum();
        if (sum != wmtk::Rational(0)) {
            bc_rational /= sum;
        }

        cgal_autorefine_demo::SampledPointInputRational sampled_pt;
        sampled_pt.tet_index = tet_id;
        sampled_pt.barycentric = bc_rational;
        surface_vertex_to_sampled[i] = static_cast<int>(sampled_points.size());
        sampled_points.push_back(sampled_pt);
    }

    if (missing_vertices > 0) {
        std::cerr << "Warning: " << missing_vertices
                  << " surface vertices were outside the tet mesh" << std::endl;
    }

    std::vector<Eigen::Vector3i> sampled_faces_list;
    sampled_faces_list.reserve(F_surface.rows());
    for (int i = 0; i < F_surface.rows(); ++i) {
        int v0 = F_surface(i, 0);
        int v1 = F_surface(i, 1);
        int v2 = F_surface(i, 2);
        int s0 = (v0 >= 0 && v0 < static_cast<int>(surface_vertex_to_sampled.size()))
                     ? surface_vertex_to_sampled[v0]
                     : -1;
        int s1 = (v1 >= 0 && v1 < static_cast<int>(surface_vertex_to_sampled.size()))
                     ? surface_vertex_to_sampled[v1]
                     : -1;
        int s2 = (v2 >= 0 && v2 < static_cast<int>(surface_vertex_to_sampled.size()))
                     ? surface_vertex_to_sampled[v2]
                     : -1;
        if (s0 < 0 || s1 < 0 || s2 < 0) {
            continue;
        }
        if (s0 == s1 || s1 == s2 || s0 == s2) {
            continue;
        }
        sampled_faces_list.emplace_back(s0, s1, s2);
    }

    if (sampled_faces_list.empty()) {
        std::cerr << "Warning: no valid triangles remain after filtering" << std::endl;
        return false;
    }

    sampled_faces.resize(sampled_faces_list.size(), 3);
    for (int i = 0; i < static_cast<int>(sampled_faces_list.size()); ++i) {
        sampled_faces.row(i) = sampled_faces_list[static_cast<std::size_t>(i)];
    }

    return true;
}
} // namespace

query_surface_tet_with_connectivity slice_tet_mesh_with_axis_plane(
    const Eigen::MatrixXi& T,
    const Eigen::MatrixXd& V,
    int axis,
    double constant)
{
    query_surface_tet_with_connectivity surface;
    if (T.rows() == 0 || V.rows() == 0) {
        return surface;
    }
    if (axis < 0 || axis > 2) {
        std::cerr << "slice_tet_mesh_with_axis_plane: axis must be 0 (x), 1 (y), or 2 (z)"
                  << std::endl;
        return surface;
    }

    using Rational = wmtk::Rational;
    const Rational constant_r(constant);
    std::unordered_map<int, int> vertex_point_map; // exact vertex on plane
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_point_map; // edge intersections
    std::vector<Eigen::Matrix<Rational, 3, 1>> point_positions; // cached world positions

    auto create_point = [&](const Eigen::Matrix<Rational, 4, 1>& bc_in,
                            int tet_id,
                            const Eigen::Vector4i& tv_ids,
                            const Eigen::Matrix<Rational, 4, 3>& tet_vertices_r) -> int {
        Eigen::Matrix<Rational, 4, 1> bc = bc_in;
        Rational sum = bc.sum();
        if (sum != Rational(0)) {
            bc /= sum;
        }

        query_point_tet_r qp;
        qp.t_id = tet_id;
        qp.tv_ids = tv_ids;
        qp.bc = bc;
        surface.points.push_back(qp);

        point_positions.push_back(barycentric_to_world_tet<Rational>(bc, tet_vertices_r));
        return static_cast<int>(surface.points.size()) - 1;
    };

    auto get_vertex_point = [&](int global_vid,
                                int local_idx,
                                int tet_id,
                                const Eigen::Vector4i& tv_ids,
                                const Eigen::Matrix<Rational, 4, 3>& tet_vertices_r) -> int {
        auto it = vertex_point_map.find(global_vid);
        if (it != vertex_point_map.end()) {
            return it->second;
        }
        Eigen::Matrix<Rational, 4, 1> bc = Eigen::Matrix<Rational, 4, 1>::Zero();
        bc(local_idx) = Rational(1);
        int pid = create_point(bc, tet_id, tv_ids, tet_vertices_r);
        vertex_point_map.emplace(global_vid, pid);
        return pid;
    };

    auto get_edge_point = [&](const EdgeKey& key,
                              int local_i,
                              int local_j,
                              const Rational& t,
                              int tet_id,
                              const Eigen::Vector4i& tv_ids,
                              const Eigen::Matrix<Rational, 4, 3>& tet_vertices_r) -> int {
        auto it = edge_point_map.find(key);
        if (it != edge_point_map.end()) {
            return it->second;
        }
        Eigen::Matrix<Rational, 4, 1> bc = Eigen::Matrix<Rational, 4, 1>::Zero();
        bc(local_i) = Rational(1) - t;
        bc(local_j) = t;
        int pid = create_point(bc, tet_id, tv_ids, tet_vertices_r);
        edge_point_map.emplace(key, pid);
        return pid;
    };

    const std::array<int, 2> uv_axes = {[&]() -> std::array<int, 2> {
        if (axis == 0) return {1, 2}; // slice plane x = constant -> use y,z for ordering
        if (axis == 1) return {0, 2}; // y = constant -> use x,z
        return {0, 1}; // z = constant -> use x,y
    }()};

    for (int tet_id = 0; tet_id < T.rows(); ++tet_id) {
        Eigen::Vector4i tv_ids = T.row(tet_id);
        Eigen::Matrix<Rational, 4, 3> tet_vertices_r;
        for (int i = 0; i < 4; ++i) {
            tet_vertices_r.row(i) = V.row(tv_ids(i)).unaryExpr([](double v) { return Rational(v); });
        }

        Rational values[4];
        for (int i = 0; i < 4; ++i) {
            values[i] = tet_vertices_r(i, axis) - constant_r;
        }

        std::vector<int> tet_point_ids;
        auto add_point_to_polygon = [&](int pid) {
            if (std::find(tet_point_ids.begin(), tet_point_ids.end(), pid) == tet_point_ids.end()) {
                tet_point_ids.push_back(pid);
            }
        };

        // Add vertices that lie on the plane exactly
        for (int i = 0; i < 4; ++i) {
            if (values[i] == Rational(0)) {
                add_point_to_polygon(get_vertex_point(tv_ids(i), i, tet_id, tv_ids, tet_vertices_r));
            }
        }

        // Add edge intersection points
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                Rational v0 = values[i];
                Rational v1 = values[j];

                // Edge lies on plane: keep both endpoints to capture co-planar faces
                if (v0 == Rational(0) && v1 == Rational(0)) {
                    add_point_to_polygon(get_vertex_point(tv_ids(i), i, tet_id, tv_ids, tet_vertices_r));
                    add_point_to_polygon(get_vertex_point(tv_ids(j), j, tet_id, tv_ids, tet_vertices_r));
                    continue;
                }

                // Proper crossing
                if ((v0 > Rational(0) && v1 < Rational(0)) || (v0 < Rational(0) && v1 > Rational(0))) {
                    Rational t = v0 / (v0 - v1); // exact parameter
                    add_point_to_polygon(get_edge_point(
                        make_edge_key(tv_ids(i), tv_ids(j)),
                        i,
                        j,
                        t,
                        tet_id,
                        tv_ids,
                        tet_vertices_r));
                }
            }
        }

        if (tet_point_ids.size() < 3) {
            continue; // intersection is a segment or a point
        }

        // Order polygon vertices around centroid in the slicing plane
        Eigen::Matrix<Rational, 2, 1> centroid = Eigen::Matrix<Rational, 2, 1>::Zero();
        for (int pid : tet_point_ids) {
            const auto& p = point_positions[pid];
            centroid(0) += p(uv_axes[0]);
            centroid(1) += p(uv_axes[1]);
        }
        centroid /= Rational(static_cast<int>(tet_point_ids.size()));

        auto vector_from_centroid = [&](int pid) {
            const auto& p = point_positions[pid];
            return Eigen::Matrix<Rational, 2, 1>(
                p(uv_axes[0]) - centroid(0),
                p(uv_axes[1]) - centroid(1));
        };
        auto is_upper_half = [&](const Eigen::Matrix<Rational, 2, 1>& v) {
            return v(1) > Rational(0) || (v(1) == Rational(0) && v(0) >= Rational(0));
        };

        std::sort(tet_point_ids.begin(), tet_point_ids.end(), [&](int a, int b) {
            const auto va = vector_from_centroid(a);
            const auto vb = vector_from_centroid(b);

            const bool upper_a = is_upper_half(va);
            const bool upper_b = is_upper_half(vb);
            if (upper_a != upper_b) {
                return upper_a; // upper half-plane first
            }

            const Rational cross = va(0) * vb(1) - va(1) * vb(0);
            if (cross != Rational(0)) {
                return cross > Rational(0); // counterclockwise order
            }

            // Colinear with centroid: use distance as deterministic tie-breaker
            const Rational dist2_a = va.squaredNorm();
            const Rational dist2_b = vb.squaredNorm();
            if (dist2_a != dist2_b) {
                return dist2_a < dist2_b;
            }

            return a < b; // final tie-breaker to maintain strict weak ordering
        });

        // Ensure consistent orientation (normal aligned with +axis where possible)
        if (tet_point_ids.size() >= 3) {
            const auto& p0 = point_positions[tet_point_ids[0]];
            const auto& p1 = point_positions[tet_point_ids[1]];
            const auto& p2 = point_positions[tet_point_ids[2]];
            Eigen::Matrix<Rational, 3, 1> orient_vec = (p1 - p0).cross(p2 - p0);
            if (orient_vec(axis) < Rational(0)) {
                std::reverse(tet_point_ids.begin() + 1, tet_point_ids.end());
            }
        }

        // Triangulate polygon as a fan
        for (size_t i = 1; i + 1 < tet_point_ids.size(); ++i) {
            surface.query_triangles.emplace_back(
                tet_point_ids[0],
                tet_point_ids[i],
                tet_point_ids[i + 1]);
            surface.tet_ids.push_back(tet_id);
        }
    }

    return surface;
}

query_surface_tet sample_query_surface_large_triangle(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out)
{
    // Create input surface mesh similar to main_arrangement.cpp
    std::vector<double> in_coords;
    std::vector<uint> in_tris;
    std::vector<uint> in_labels;

    // Convert V_out to in_coords (flatten the matrix)
    for (int i = 0; i < V_out.rows(); i++) {
        for (int j = 0; j < V_out.cols(); j++) {
            in_coords.push_back(V_out(i, j));
        }
    }

    // Convert T_out to in_tris, adding all four triangles for each tetrahedron
    for (int t_id = 0; t_id < T_out.rows(); t_id++) {
        auto tet = T_out.row(t_id);
        if (tet.size() >= 4) {
            // Extract the four faces of the tetrahedron
            std::vector<std::vector<int>> faces = {
                {tet[0], tet[1], tet[2]},
                {tet[0], tet[1], tet[3]},
                {tet[0], tet[2], tet[3]},
                {tet[1], tet[2], tet[3]}};

            for (const auto& tri : faces) {
                for (const auto& vertex_id : tri) {
                    in_tris.push_back(static_cast<uint>(vertex_id));
                }
                // Set label based on tetrahedron ID
                in_labels.push_back(static_cast<uint>(t_id));
            }
        }
    }

    // Randomly sample three points from T_out to create a label 1 triangle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::set<int>> vertex_to_labels(V_out.rows() + 3);

    if (T_out.rows() > 0) {
        std::uniform_int_distribution<> tet_dis(0, T_out.rows() - 1);
        std::uniform_real_distribution<> bary_dis(0.0, 1.0);

        std::vector<std::vector<double>> sampled_points;

        // Sample three points
        for (int sample = 0; sample < 3; sample++) {
            // Randomly select a tetrahedron
            int tet_id = tet_dis(gen);
            auto tet = T_out.row(tet_id);

            if (tet.size() >= 4) {
                // Generate random barycentric coordinates
                double r1 = bary_dis(gen);
                double r2 = bary_dis(gen);
                double r3 = bary_dis(gen);
                double r4 = bary_dis(gen);

                // Normalize to ensure they sum to 1
                double sum = r1 + r2 + r3 + r4;
                r1 /= sum;
                r2 /= sum;
                r3 /= sum;
                r4 /= sum;

                // Get vertices of the tetrahedron
                auto v0 = V_out.row(tet[0]);
                auto v1 = V_out.row(tet[1]);
                auto v2 = V_out.row(tet[2]);
                auto v3 = V_out.row(tet[3]);

                // Interpolate position using barycentric coordinates
                std::vector<double> point(3, 0.0);
                for (int i = 0; i < 3; i++) {
                    point[i] = r1 * v0[i] + r2 * v1[i] + r3 * v2[i] + r4 * v3[i];
                }

                sampled_points.push_back(point);
                vertex_to_labels[V_out.rows() + sample].insert(tet_id);
            }
        }

        // Add sampled points to coordinates and create triangle
        if (sampled_points.size() == 3) {
            uint start_vertex_id = V_out.rows();

            // Add sampled points to in_coords
            for (const auto& point : sampled_points) {
                for (const auto& coord : point) {
                    in_coords.push_back(coord);
                }
            }

            // Add triangle with label
            in_tris.insert(
                in_tris.end(),
                {start_vertex_id, start_vertex_id + 1, start_vertex_id + 2});
            in_labels.push_back(T_out.rows());
        }
    }

    // init the necessary data structures
    point_arena arena;
    std::vector<genericPoint*> arr_verts;
    std::vector<uint> arr_in_tris, arr_out_tris;
    std::vector<std::bitset<NBIT>> arr_in_labels;
    std::vector<DuplTriInfo> dupl_triangles;
    Labels labels;
    cinolib::Octree octree;

    // arrangement, last parameter is false to avoid parallelization
    customArrangementPipeline(
        in_coords,
        in_tris,
        in_labels,
        arr_in_tris,
        arr_in_labels,
        arena,
        arr_verts,
        arr_out_tris,
        labels,
        octree,
        dupl_triangles,
        false);

    // create FastTrimesh
    FastTrimesh tm(arr_verts, arr_out_tris, true);
    // Prepare output data
    std::vector<double> out_coords;
    std::vector<uint> out_tris;
    std::vector<std::bitset<NBIT>> out_labels;
    {
        tm.resetTrianglesInfo();
        uint num_tris = 0;

        // All triangles
        for (uint t_id = 0; t_id < tm.numTris(); t_id++) {
            tm.setTriInfo(t_id, 1);
            num_tris++;
        }

        getFinalMeshInOder(tm, labels, num_tris, out_coords, out_tris, out_labels);
    }

    // get barycentric coordinates of the output triangles
    std::vector<int> out_tri_ids;
    vertex_to_labels.resize(tm.numVerts());
    for (uint t_id = 0; t_id < tm.numTris(); t_id++) {
        uint v0 = tm.tri(t_id)[0];
        uint v1 = tm.tri(t_id)[1];
        uint v2 = tm.tri(t_id)[2];

        for (uint label_id = 0; label_id < labels.num; label_id++) {
            if (labels.surface[t_id][label_id]) {
                vertex_to_labels[v0].insert(label_id);
                vertex_to_labels[v1].insert(label_id);
                vertex_to_labels[v2].insert(label_id);
            }
        }

        if (labels.surface[t_id][labels.num - 1]) {
            out_tri_ids.push_back(t_id);
        }
    }

    std::cout << "out_tri_ids.size(): " << out_tri_ids.size() << std::endl;
    query_surface_tet query_surface;
    for (int i = 0; i < out_tri_ids.size(); i++) {
        int triangle_id = out_tri_ids[i];
        std::cout << "checking triangle id: " << triangle_id << std::endl;
        uint v0_idx = tm.tri(triangle_id)[0];
        uint v1_idx = tm.tri(triangle_id)[1];
        uint v2_idx = tm.tri(triangle_id)[2];
        int containing_tet_id = -1;
        for (int tet_id = 0; tet_id < T_out.rows(); tet_id++) {
            if (vertex_to_labels[v0_idx].count(tet_id) && vertex_to_labels[v1_idx].count(tet_id) &&
                vertex_to_labels[v2_idx].count(tet_id)) {
                std::cout << "triangle " << triangle_id << " is in tet " << tet_id << std::endl;
                containing_tet_id = tet_id;
                break;
            }
        }
        if (containing_tet_id == -1) {
            std::cout << "ERRRO! triangle " << triangle_id << " is not in any tet" << std::endl;
            exit(1);
        }
        query_triangle_tet q_tri;
        q_tri.t_id = containing_tet_id;
        q_tri.tv_ids = T_out.row(containing_tet_id);
        auto v0_world = Eigen::Vector3d(
            out_coords[v0_idx * 3],
            out_coords[v0_idx * 3 + 1],
            out_coords[v0_idx * 3 + 2]);
        auto v1_world = Eigen::Vector3d(
            out_coords[v1_idx * 3],
            out_coords[v1_idx * 3 + 1],
            out_coords[v1_idx * 3 + 2]);
        auto v2_world = Eigen::Vector3d(
            out_coords[v2_idx * 3],
            out_coords[v2_idx * 3 + 1],
            out_coords[v2_idx * 3 + 2]);

        Eigen::Matrix<double, 4, 3> tet_Vs;
        tet_Vs.row(0) = V_out.row(T_out(containing_tet_id, 0));
        tet_Vs.row(1) = V_out.row(T_out(containing_tet_id, 1));
        tet_Vs.row(2) = V_out.row(T_out(containing_tet_id, 2));
        tet_Vs.row(3) = V_out.row(T_out(containing_tet_id, 3));
        q_tri.bcs[0] = world_to_barycentric_tet(v0_world, tet_Vs);
        q_tri.bcs[1] = world_to_barycentric_tet(v1_world, tet_Vs);
        q_tri.bcs[2] = world_to_barycentric_tet(v2_world, tet_Vs);
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                if (std::abs(q_tri.bcs[j](k)) < 1e-15) {
                    q_tri.bcs[j](k) = 0.0;
                }
            }
        }
        query_surface.triangles.push_back(q_tri);
    }
    return query_surface;
}

query_surface_tet_with_connectivity sample_query_surface_tet_with_connectivity(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out)
{
    std::cout
        << "Sampling query surface with connectivity using autorefine_sampled_triangles_rational..."
        << std::endl;
    using MatrixXr = Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector3r = Eigen::Matrix<wmtk::Rational, 3, 1>;
    using Vector4r = Eigen::Matrix<wmtk::Rational, 4, 1>;
    if (T_out.rows() == 0) {
        return query_surface_tet_with_connectivity();
    }
    // Convert V_out from double to rational
    MatrixXr V_rational(V_out.rows(), V_out.cols());
    for (int i = 0; i < V_out.rows(); i++) {
        for (int j = 0; j < V_out.cols(); j++) {
            V_rational(i, j) = wmtk::Rational(V_out(i, j));
        }
    }
    // Randomly sample three points from T_out to create a query triangle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> tet_dis(0, T_out.rows() - 1);
    std::uniform_real_distribution<> bary_dis(0.0, 1.0);
    std::vector<cgal_autorefine_demo::SampledPointInputRational> sampled_points;
    sampled_points.reserve(3);
    // Sample three points
    for (int sample = 0; sample < 3; sample++) {
        // Randomly select a tetrahedron
        int tet_id = tet_dis(gen);
        auto tet = T_out.row(tet_id);
        if (tet.size() >= 4) {
            // Generate random barycentric coordinates
            double r1 = bary_dis(gen);
            double r2 = bary_dis(gen);
            double r3 = bary_dis(gen);
            double r4 = bary_dis(gen);
            // Normalize to ensure they sum to 1
            double sum = r1 + r2 + r3 + r4;
            r1 /= sum;
            r2 /= sum;
            r3 /= sum;
            r4 /= sum;
            // Create SampledPointInputRational
            cgal_autorefine_demo::SampledPointInputRational sampled_pt;
            sampled_pt.tet_index = tet_id;
            sampled_pt.barycentric = Vector4r(
                wmtk::Rational(r1),
                wmtk::Rational(r2),
                wmtk::Rational(r3),
                wmtk::Rational(r4));
            sampled_points.push_back(sampled_pt);
        }
    }

    // Create sampled_faces matrix with one triangle (indices 0, 1, 2 for the 3 sampled points)
    Eigen::MatrixXi sampled_faces(1, 3);
    sampled_faces << 0, 1, 2;
    // Call autorefine_sampled_triangles_rational
    std::cout << "Calling autorefine_sampled_triangles_rational..." << std::endl;
    cgal_autorefine_demo::AutorefineResultRational autorefine_result =
        cgal_autorefine_demo::autorefine_sampled_triangles_rational(
            V_rational,
            T_out,
            sampled_points,
            sampled_faces);
    std::cout << "Autorefine completed: " << autorefine_result.refined_points.size()
              << " refined points, " << autorefine_result.refined_triangles.size()
              << " refined triangles" << std::endl;
    std::cout << "Sampled fragment triangles: "
              << autorefine_result.sampled_fragment_triangles.size() << std::endl;
    // Process the result to build query_surface_tet_with_connectivity
    query_surface_tet_with_connectivity query_surface;
    if (autorefine_result.sampled_fragment_triangles.empty()) {
        std::cout << "No refined sampled triangles found" << std::endl;
        return query_surface;
    }
    // Build mapping from refined_points index to query_surface.points index
    std::map<std::size_t, int> refined_point_to_surface_point;
    // Process all vertices used in sampled_fragment_triangles
    std::set<std::size_t> refined_vertex_ids_used;
    for (const auto& tri : autorefine_result.sampled_fragment_triangles) {
        refined_vertex_ids_used.insert(tri[0]);
        refined_vertex_ids_used.insert(tri[1]);
        refined_vertex_ids_used.insert(tri[2]);
    }
    // For each vertex used in refined triangles, create a query_point_tet_r
    for (std::size_t refined_v_id : refined_vertex_ids_used) {
        // Get the point's position
        const cgal_autorefine_demo::RationalPoint& p =
            autorefine_result.refined_points[refined_v_id];
        Vector3r point_pos;
        point_pos(0) = wmtk::Rational(p.x(), false);
        point_pos(1) = wmtk::Rational(p.y(), false);
        point_pos(2) = wmtk::Rational(p.z(), false);
        // Find which tet this point belongs to
        int local_tet_id = -1;
        if (refined_v_id < autorefine_result.vertex_tet_sets.size()) {
            const auto& tet_set = autorefine_result.vertex_tet_sets[refined_v_id];
            if (!tet_set.empty()) {
                local_tet_id = *tet_set.begin();
            }
        }
        // Fallback: try to get from sampled_fragment_tet_ids
        if (local_tet_id == -1) {
            for (std::size_t tri_idx = 0;
                 tri_idx < autorefine_result.sampled_fragment_triangles.size();
                 ++tri_idx) {
                const auto& tri = autorefine_result.sampled_fragment_triangles[tri_idx];
                if (tri[0] == refined_v_id || tri[1] == refined_v_id || tri[2] == refined_v_id) {
                    local_tet_id = autorefine_result.sampled_fragment_tet_ids(tri_idx);
                    break;
                }
            }
        }
        if (local_tet_id == -1 || local_tet_id >= T_out.rows()) {
            std::cerr << "Warning: Could not find valid tet_id for point " << refined_v_id
                      << std::endl;
            continue;
        }
        // Get tet vertices
        Eigen::Vector4i tv_ids = T_out.row(local_tet_id);
        Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices;
        for (int i = 0; i < 4; ++i) {
            tet_vertices.row(i) = V_rational.row(tv_ids(i));
        }
        // Compute barycentric coordinates
        Vector4r barycentric_coords =
            world_to_barycentric_tet<wmtk::Rational>(point_pos, tet_vertices);
        // Normalize barycentric coordinates
        wmtk::Rational sum = barycentric_coords(0) + barycentric_coords(1) + barycentric_coords(2) +
                             barycentric_coords(3);
        if (sum != wmtk::Rational(0)) {
            barycentric_coords = barycentric_coords / sum;
        }
        // Clean up small values
        for (int bc_idx = 0; bc_idx < 4; ++bc_idx) {
            if (std::abs(barycentric_coords(bc_idx).to_double()) < 1e-14) {
                barycentric_coords(bc_idx) = wmtk::Rational(0);
            }
        }
        // Create query_point_tet_r
        query_point_tet_r qp;
        qp.t_id = local_tet_id;
        qp.bc = barycentric_coords;
        qp.tv_ids = tv_ids;
        // Add to points list
        int point_idx = query_surface.points.size();
        query_surface.points.push_back(qp);
        refined_point_to_surface_point[refined_v_id] = point_idx;
    }
    // Process sampled_fragment_triangles to build query_triangles
    for (std::size_t i = 0; i < autorefine_result.sampled_fragment_triangles.size(); ++i) {
        const cgal_autorefine_demo::Triangle& refined_tri =
            autorefine_result.sampled_fragment_triangles[i];
        // Map refined_points indices to surface.points indices
        Eigen::Vector3i new_tri;
        bool all_mapped = true;
        for (int corner = 0; corner < 3; ++corner) {
            std::size_t refined_v_id = refined_tri[corner];
            auto it = refined_point_to_surface_point.find(refined_v_id);
            if (it != refined_point_to_surface_point.end()) {
                new_tri(corner) = it->second;
            } else {
                std::cerr << "Warning: Could not map refined vertex " << refined_v_id
                          << " to surface point" << std::endl;
                all_mapped = false;
                break;
            }
        }
        if (!all_mapped) {
            continue;
        }
        // Add the new triangle
        query_surface.query_triangles.push_back(new_tri);
        // Add corresponding tet_id
        int local_tet_id = autorefine_result.sampled_fragment_tet_ids(i);
        query_surface.tet_ids.push_back(local_tet_id);
    }
    std::cout << "Created surface with " << query_surface.points.size() << " unique points and "
              << query_surface.query_triangles.size() << " triangles" << std::endl;
    return query_surface;
}

query_surface_tet_with_connectivity query_surface_tet_with_connectivity_from_triangle_mesh(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    double tolerance,
    bool verbose)
{
    query_surface_tet_with_connectivity query_surface;
    if (T_out.rows() == 0 || V_out.rows() == 0 || F_surface.rows() == 0) {
        return query_surface;
    }

    using MatrixXr = Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector4r = Eigen::Matrix<wmtk::Rational, 4, 1>;

    std::cout << "Building query surface with connectivity from triangle mesh..." << std::endl;
    std::cout << "  Surface vertices: " << V_surface.rows() << std::endl;
    std::cout << "  Surface triangles: " << F_surface.rows() << std::endl;

    MatrixXr V_rational(V_out.rows(), V_out.cols());
    for (int i = 0; i < V_out.rows(); i++) {
        for (int j = 0; j < V_out.cols(); j++) {
            V_rational(i, j) = wmtk::Rational(V_out(i, j));
        }
    }

    std::vector<cgal_autorefine_demo::SampledPointInputRational> sampled_points;
    Eigen::MatrixXi sampled_faces;
    if (!build_sampled_points_and_faces(
            V_out,
            T_out,
            V_surface,
            F_surface,
            tolerance,
            verbose,
            sampled_points,
            sampled_faces)) {
        return query_surface;
    }

    std::cout << "Calling autorefine_sampled_triangles_rational..." << std::endl;
    cgal_autorefine_demo::AutorefineResultRational autorefine_result =
        cgal_autorefine_demo::autorefine_sampled_triangles_rational(
            V_rational,
            T_out,
            sampled_points,
            sampled_faces,
            verbose);
    std::cout << "Autorefine completed: " << autorefine_result.refined_points.size()
              << " refined points, " << autorefine_result.refined_triangles.size()
              << " refined triangles" << std::endl;
    std::cout << "Sampled fragment triangles: "
              << autorefine_result.sampled_fragment_triangles.size() << std::endl;

    if (autorefine_result.sampled_fragment_triangles.empty()) {
        std::cout << "No refined sampled triangles found" << std::endl;
        return query_surface;
    }

    std::map<std::size_t, int> refined_point_to_surface_point;
    std::set<std::size_t> refined_vertex_ids_used;
    for (const auto& tri : autorefine_result.sampled_fragment_triangles) {
        refined_vertex_ids_used.insert(tri[0]);
        refined_vertex_ids_used.insert(tri[1]);
        refined_vertex_ids_used.insert(tri[2]);
    }

    for (std::size_t refined_v_id : refined_vertex_ids_used) {
        const cgal_autorefine_demo::RationalPoint& p =
            autorefine_result.refined_points[refined_v_id];
        Eigen::Matrix<wmtk::Rational, 3, 1> point_pos;
        point_pos(0) = wmtk::Rational(p.x(), false);
        point_pos(1) = wmtk::Rational(p.y(), false);
        point_pos(2) = wmtk::Rational(p.z(), false);

        int local_tet_id = -1;
        if (refined_v_id < autorefine_result.vertex_tet_sets.size()) {
            const auto& tet_set = autorefine_result.vertex_tet_sets[refined_v_id];
            if (!tet_set.empty()) {
                local_tet_id = *tet_set.begin();
            }
        }
        if (local_tet_id == -1) {
            for (std::size_t tri_idx = 0;
                 tri_idx < autorefine_result.sampled_fragment_triangles.size();
                 ++tri_idx) {
                const auto& tri = autorefine_result.sampled_fragment_triangles[tri_idx];
                if (tri[0] == refined_v_id || tri[1] == refined_v_id || tri[2] == refined_v_id) {
                    local_tet_id = autorefine_result.sampled_fragment_tet_ids(tri_idx);
                    break;
                }
            }
        }
        if (local_tet_id == -1 || local_tet_id >= T_out.rows()) {
            if (verbose) {
                std::cerr << "Warning: Could not find valid tet_id for point " << refined_v_id
                          << std::endl;
            }
            continue;
        }

        Eigen::Vector4i tv_ids = T_out.row(local_tet_id);
        Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices;
        for (int i = 0; i < 4; ++i) {
            tet_vertices.row(i) = V_rational.row(tv_ids(i));
        }

        Vector4r barycentric_coords =
            world_to_barycentric_tet<wmtk::Rational>(point_pos, tet_vertices);
        wmtk::Rational sum = barycentric_coords.sum();
        if (sum != wmtk::Rational(0)) {
            barycentric_coords = barycentric_coords / sum;
        }
        for (int bc_idx = 0; bc_idx < 4; ++bc_idx) {
            if (std::abs(barycentric_coords(bc_idx).to_double()) < 1e-14) {
                barycentric_coords(bc_idx) = wmtk::Rational(0);
            }
        }

        query_point_tet_r qp;
        qp.t_id = local_tet_id;
        qp.bc = barycentric_coords;
        qp.tv_ids = tv_ids;

        int point_idx = static_cast<int>(query_surface.points.size());
        query_surface.points.push_back(qp);
        refined_point_to_surface_point[refined_v_id] = point_idx;
    }

    for (std::size_t i = 0; i < autorefine_result.sampled_fragment_triangles.size(); ++i) {
        const cgal_autorefine_demo::Triangle& refined_tri =
            autorefine_result.sampled_fragment_triangles[i];
        Eigen::Vector3i new_tri;
        bool all_mapped = true;
        for (int corner = 0; corner < 3; ++corner) {
            std::size_t refined_v_id = refined_tri[corner];
            auto it = refined_point_to_surface_point.find(refined_v_id);
            if (it != refined_point_to_surface_point.end()) {
                new_tri(corner) = it->second;
            } else {
                all_mapped = false;
                break;
            }
        }
        if (!all_mapped) {
            continue;
        }
        query_surface.query_triangles.push_back(new_tri);
        int local_tet_id = autorefine_result.sampled_fragment_tet_ids(i);
        query_surface.tet_ids.push_back(local_tet_id);
    }

    std::cout << "Created surface with " << query_surface.points.size() << " unique points and "
              << query_surface.query_triangles.size() << " triangles" << std::endl;
    return query_surface;
}

bool arrangement_triangle_mesh_in_tet_mesh(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    Eigen::MatrixXd& V_arranged,
    Eigen::MatrixXi& F_arranged,
    double tolerance,
    bool verbose)
{
    V_arranged.resize(0, 3);
    F_arranged.resize(0, 3);
    if (T_out.rows() == 0 || V_out.rows() == 0 || F_surface.rows() == 0) {
        return false;
    }
    (void)tolerance;
    (void)verbose;

    std::cout << "Arranging triangle mesh with tet boundaries..." << std::endl;
    std::cout << "  Surface vertices: " << V_surface.rows() << std::endl;
    std::cout << "  Surface triangles: " << F_surface.rows() << std::endl;

    std::vector<cgal_autorefine_demo::Point> points;
    points.reserve(static_cast<std::size_t>(V_out.rows() + V_surface.rows()));
    for (int i = 0; i < V_out.rows(); ++i) {
        points.emplace_back(V_out(i, 0), V_out(i, 1), V_out(i, 2));
    }
    const std::size_t surface_offset = points.size();
    for (int i = 0; i < V_surface.rows(); ++i) {
        points.emplace_back(V_surface(i, 0), V_surface(i, 1), V_surface(i, 2));
    }

    auto tet_triangles = cgal_autorefine_demo::extract_all_tet_triangles(T_out);
    std::vector<cgal_autorefine_demo::Triangle> triangles;
    std::vector<bool> is_surface_triangle;
    triangles.reserve(tet_triangles.size() + static_cast<std::size_t>(F_surface.rows()));
    is_surface_triangle.reserve(triangles.capacity());

    for (const auto& tet_tri : tet_triangles) {
        triangles.push_back(tet_tri.triangle);
        is_surface_triangle.push_back(false);
    }

    for (int i = 0; i < F_surface.rows(); ++i) {
        cgal_autorefine_demo::Triangle tri{};
        tri[0] = surface_offset + static_cast<std::size_t>(F_surface(i, 0));
        tri[1] = surface_offset + static_cast<std::size_t>(F_surface(i, 1));
        tri[2] = surface_offset + static_cast<std::size_t>(F_surface(i, 2));
        triangles.push_back(tri);
        is_surface_triangle.push_back(true);
    }

    std::vector<std::vector<std::size_t>> working_triangles;
    working_triangles.reserve(triangles.size());
    for (const auto& tri : triangles) {
        working_triangles.push_back({tri[0], tri[1], tri[2]});
    }

    std::vector<std::size_t> triangle_source_ids;
    TriangleTrackingVisitor visitor(triangle_source_ids);
    CGAL::Polygon_mesh_processing::autorefine_triangle_soup(
        points,
        working_triangles,
        CGAL::parameters::visitor(visitor).apply_iterative_snap_rounding(true));

    const std::size_t invalid_id = static_cast<std::size_t>(-1);
    std::unordered_map<std::size_t, int> point_map;
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3i> faces;
    positions.reserve(points.size());
    faces.reserve(working_triangles.size());

    auto get_or_add_point = [&](std::size_t pid) -> int {
        auto it = point_map.find(pid);
        if (it != point_map.end()) {
            return it->second;
        }
        const auto& p = points[pid];
        Eigen::Vector3d pos;
        pos(0) = CGAL::to_double(p.x());
        pos(1) = CGAL::to_double(p.y());
        pos(2) = CGAL::to_double(p.z());
        int new_idx = static_cast<int>(positions.size());
        positions.push_back(pos);
        point_map.emplace(pid, new_idx);
        return new_idx;
    };

    for (std::size_t i = 0; i < working_triangles.size(); ++i) {
        const auto& tri = working_triangles[i];
        if (tri.size() != 3) {
            continue;
        }
        const std::size_t src_id =
            (i < triangle_source_ids.size()) ? triangle_source_ids[i] : invalid_id;
        if (src_id == invalid_id || src_id >= is_surface_triangle.size()) {
            continue;
        }
        if (!is_surface_triangle[src_id]) {
            continue;
        }
        Eigen::Vector3i out_tri;
        out_tri(0) = get_or_add_point(tri[0]);
        out_tri(1) = get_or_add_point(tri[1]);
        out_tri(2) = get_or_add_point(tri[2]);
        faces.push_back(out_tri);
    }

    if (faces.empty()) {
        std::cout << "No arranged surface triangles found" << std::endl;
        return false;
    }

    V_arranged.resize(static_cast<int>(positions.size()), 3);
    for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
        V_arranged.row(i) = positions[static_cast<std::size_t>(i)];
    }
    F_arranged.resize(static_cast<int>(faces.size()), 3);
    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        F_arranged.row(i) = faces[static_cast<std::size_t>(i)];
    }

    std::cout << "Arranged mesh: " << V_arranged.rows() << " vertices, " << F_arranged.rows()
              << " triangles" << std::endl;
    return true;
}

query_surface_tet_with_connectivity query_surface_tet_with_connectivity_no_arrangement(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    bool verbose)
{
    query_surface_tet_with_connectivity query_surface;
    if (T_out.rows() == 0 || V_out.rows() == 0 || V_surface.rows() == 0 ||
        F_surface.rows() == 0) {
        return query_surface;
    }

    std::cout << "Building query surface without arrangement..." << std::endl;
    std::cout << "  Surface vertices: " << V_surface.rows() << std::endl;
    std::cout << "  Surface triangles: " << F_surface.rows() << std::endl;

    // Precompute tet AABBs and a uniform grid accelerator (double precision).
    std::vector<TetAabb> tet_aabbs(T_out.rows());
    Eigen::Vector3d global_min = V_out.row(0).transpose();
    Eigen::Vector3d global_max = V_out.row(0).transpose();
    for (int i = 0; i < V_out.rows(); ++i) {
        Eigen::Vector3d v = V_out.row(i).transpose();
        global_min = global_min.cwiseMin(v);
        global_max = global_max.cwiseMax(v);
    }

    for (int t = 0; t < T_out.rows(); ++t) {
        Eigen::Vector4i tv = T_out.row(t);
        Eigen::Vector3d tmin = V_out.row(tv(0)).transpose();
        Eigen::Vector3d tmax = V_out.row(tv(0)).transpose();
        for (int j = 1; j < 4; ++j) {
            Eigen::Vector3d v = V_out.row(tv(j)).transpose();
            tmin = tmin.cwiseMin(v);
            tmax = tmax.cwiseMax(v);
        }
        tet_aabbs[t] = TetAabb{tmin, tmax};
    }

    Eigen::Vector3d extents = (global_max - global_min).cwiseMax(1e-12);
    double max_extent = extents.maxCoeff();
    int base = std::max(1, static_cast<int>(std::ceil(std::cbrt(static_cast<double>(T_out.rows())))));
    Eigen::Vector3d scale = Eigen::Vector3d::Ones();
    if (max_extent > 0.0) {
        scale = extents / max_extent;
    }
    Eigen::Vector3i grid_dims(
        std::max(1, static_cast<int>(std::round(base * scale(0)))),
        std::max(1, static_cast<int>(std::round(base * scale(1)))),
        std::max(1, static_cast<int>(std::round(base * scale(2)))));

    Eigen::Vector3d cell_size(
        extents(0) / grid_dims(0),
        extents(1) / grid_dims(1),
        extents(2) / grid_dims(2));

    const int grid_size = grid_dims(0) * grid_dims(1) * grid_dims(2);
    std::vector<std::vector<int>> grid_cells(grid_size);

    auto clamp_cell = [&](double value, int axis) -> int {
        if (grid_dims(axis) <= 1) {
            return 0;
        }
        double t = (value - global_min(axis)) / cell_size(axis);
        int idx = static_cast<int>(std::floor(t));
        if (idx < 0) {
            idx = 0;
        } else if (idx >= grid_dims(axis)) {
            idx = grid_dims(axis) - 1;
        }
        return idx;
    };

    auto cell_index = [&](int ix, int iy, int iz) -> int {
        return (ix * grid_dims(1) + iy) * grid_dims(2) + iz;
    };

    for (int t = 0; t < T_out.rows(); ++t) {
        const auto& aabb = tet_aabbs[t];
        int ix0 = clamp_cell(aabb.min(0), 0);
        int iy0 = clamp_cell(aabb.min(1), 1);
        int iz0 = clamp_cell(aabb.min(2), 2);
        int ix1 = clamp_cell(aabb.max(0), 0);
        int iy1 = clamp_cell(aabb.max(1), 1);
        int iz1 = clamp_cell(aabb.max(2), 2);
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                for (int iz = iz0; iz <= iz1; ++iz) {
                    grid_cells[cell_index(ix, iy, iz)].push_back(t);
                }
            }
        }
    }

    Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, 3> V_rational = toRationalMatrix(V_out);

    query_surface.points.reserve(V_surface.rows());
    std::vector<int> point_valid(V_surface.rows(), 0);
    std::vector<query_point_tet_r> points(V_surface.rows());
    std::vector<int> missing_flags(V_surface.rows(), 0);

    auto point_in_tet = [&](const Eigen::Matrix<wmtk::Rational, 3, 1>& p,
                            int tet_id,
                            Eigen::Matrix<wmtk::Rational, 4, 1>& bc_out) -> bool {
        Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices;
        Eigen::Vector4i tv_ids = T_out.row(tet_id);
        for (int j = 0; j < 4; ++j) {
            tet_vertices.row(j) = V_rational.row(tv_ids(j));
        }
        bc_out = world_to_barycentric_tet<wmtk::Rational>(p, tet_vertices);
        wmtk::Rational sum = bc_out.sum();
        if (sum != wmtk::Rational(0)) {
            bc_out /= sum;
        }
        for (int j = 0; j < 4; ++j) {
            if (bc_out(j) < wmtk::Rational(0) || bc_out(j) > wmtk::Rational(1)) {
                return false;
            }
        }
        return true;
    };

    igl::parallel_for(
        V_surface.rows(),
        [&](int i) {
            Eigen::Vector3d p_double = V_surface.row(i);
            Eigen::Matrix<wmtk::Rational, 3, 1> p = toRationalVector(p_double);

            int tet_id = -1;
            Eigen::Matrix<wmtk::Rational, 4, 1> bc_rational =
                Eigen::Matrix<wmtk::Rational, 4, 1>::Zero();

            int ix = clamp_cell(p_double(0), 0);
            int iy = clamp_cell(p_double(1), 1);
            int iz = clamp_cell(p_double(2), 2);
            const auto& candidates = grid_cells[cell_index(ix, iy, iz)];

            for (int cand : candidates) {
                if (point_in_tet(p, cand, bc_rational)) {
                    tet_id = cand;
                    break;
                }
            }

            if (tet_id < 0) {
                auto fallback = findTetContainingPointRational(V_rational, T_out, p);
                tet_id = fallback.first;
                bc_rational = fallback.second;
            }

            query_point_tet_r qp;
            qp.t_id = tet_id;
            if (tet_id >= 0) {
                qp.bc = bc_rational;
                qp.tv_ids = T_out.row(tet_id);
                point_valid[i] = 1;
            } else {
                qp.bc = Eigen::Matrix<wmtk::Rational, 4, 1>::Zero();
                qp.tv_ids = Eigen::Vector4i(-1, -1, -1, -1);
                missing_flags[i] = 1;
            }
            points[i] = qp;
        },
        256);

    int missing_vertices = 0;
    for (int i = 0; i < static_cast<int>(missing_flags.size()); ++i) {
        if (missing_flags[i]) {
            missing_vertices++;
            if (verbose) {
                std::cerr << "Warning: surface vertex " << i
                          << " not found in any tet (marking invalid)" << std::endl;
            }
        }
    }

    if (missing_vertices > 0) {
        std::cerr << "Warning: " << missing_vertices
                  << " surface vertices were outside the tet mesh" << std::endl;
    }

    query_surface.points = std::move(points);

    query_surface.query_triangles.reserve(F_surface.rows());
    query_surface.tet_ids.reserve(F_surface.rows());
    int skipped_triangles = 0;
    for (int i = 0; i < F_surface.rows(); ++i) {
        int v0 = F_surface(i, 0);
        int v1 = F_surface(i, 1);
        int v2 = F_surface(i, 2);
        if (v0 < 0 || v1 < 0 || v2 < 0 || v0 >= V_surface.rows() || v1 >= V_surface.rows() ||
            v2 >= V_surface.rows()) {
            skipped_triangles++;
            continue;
        }
        if (!point_valid[v0] || !point_valid[v1] || !point_valid[v2]) {
            skipped_triangles++;
            continue;
        }
        query_surface.query_triangles.emplace_back(v0, v1, v2);

        int t0 = query_surface.points[v0].t_id;
        int t1 = query_surface.points[v1].t_id;
        int t2 = query_surface.points[v2].t_id;
        if (t0 == t1 && t1 == t2) {
            query_surface.tet_ids.push_back(t0);
        } else {
            query_surface.tet_ids.push_back(-1);
        }
    }

    if (skipped_triangles > 0) {
        std::cerr << "Warning: skipped " << skipped_triangles
                  << " triangles due to invalid vertices" << std::endl;
    }

    std::cout << "Created surface with " << query_surface.points.size() << " points and "
              << query_surface.query_triangles.size() << " triangles" << std::endl;
    return query_surface;
}

query_surface_tet sample_query_surface_sub_surface(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out)
{
    query_surface_tet query_surface;
    if (T_out.rows() == 0) return query_surface;

    std::unordered_set<int> visited_tets;
    std::queue<int> tet_queue;
    std::unordered_set<std::string> added_triangles;

    tet_queue.push(0);
    visited_tets.insert(0);

    int max_tets_to_sample = std::min(20, (int)T_out.rows());
    int sampled_count = 0;

    while (!tet_queue.empty() && sampled_count < max_tets_to_sample) {
        int current_tet = tet_queue.front();
        tet_queue.pop();

        std::vector<std::vector<int>> face_vertices = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};

        for (int face_id = 0; face_id < 4; face_id++) {
            std::vector<int> face_vs = face_vertices[face_id];
            std::sort(face_vs.begin(), face_vs.end());
            std::string triangle_key = std::to_string(T_out(current_tet, face_vs[0])) + "_" +
                                       std::to_string(T_out(current_tet, face_vs[1])) + "_" +
                                       std::to_string(T_out(current_tet, face_vs[2]));

            if (added_triangles.find(triangle_key) == added_triangles.end()) {
                query_triangle_tet q_tri;
                q_tri.t_id = current_tet;
                q_tri.tv_ids = T_out.row(current_tet);

                for (int j = 0; j < 3; j++) {
                    q_tri.bcs[j] = Eigen::Vector4d::Zero();
                    int vertex_idx = face_vertices[face_id][j];
                    q_tri.bcs[j](vertex_idx) = 1.0;
                }

                query_surface.triangles.push_back(q_tri);
                added_triangles.insert(triangle_key);
            }
        }

        sampled_count++;

        for (int i = 0; i < T_out.rows(); i++) {
            if (visited_tets.find(i) == visited_tets.end()) {
                bool shares_edge = false;
                for (int j = 0; j < 4; j++) {
                    for (int k = j + 1; k < 4; k++) {
                        int edge_v1 = T_out(current_tet, j);
                        int edge_v2 = T_out(current_tet, k);
                        for (int m = 0; m < 4; m++) {
                            for (int n = m + 1; n < 4; n++) {
                                if ((T_out(i, m) == edge_v1 && T_out(i, n) == edge_v2) ||
                                    (T_out(i, m) == edge_v2 && T_out(i, n) == edge_v1)) {
                                    shares_edge = true;
                                    break;
                                }
                            }
                            if (shares_edge) break;
                        }
                        if (shares_edge) break;
                    }
                    if (shares_edge) break;
                }

                if (shares_edge && visited_tets.size() < max_tets_to_sample) {
                    tet_queue.push(i);
                    visited_tets.insert(i);
                }
            }
        }
    }

    return query_surface;
}

} // namespace tet_surface_sampling

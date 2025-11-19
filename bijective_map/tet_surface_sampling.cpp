#include "tet_surface_sampling.hpp"
#include <cinolib/io/write_OBJ.h>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
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
    if (sampled_points.size() != 3) {
        std::cerr << "ERROR: Failed to sample 3 points" << std::endl;
        return query_surface_tet_with_connectivity();
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

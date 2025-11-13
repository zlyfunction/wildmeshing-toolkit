#include "tet_surface_sampling.hpp"
#include <iostream>
#include <random>
#include <queue>
#include <unordered_set>
#include <cinolib/io/write_OBJ.h>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
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
    std::cout << "Sampling query surface with connectivity..." << std::endl;

    // Create input surface mesh similar to sample_query_surface_large_triangle
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

    // Randomly sample three points from T_out to create a label triangle
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
    std::vector<uint> vertex_id_map;

    // arrangement, last parameter is false to avoid parallelization
    vertex_id_map = customArrangementPipeline(
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

    // Build vertex_to_labels mapping
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

    // Now create query_surface_tet_with_connectivity
    query_surface_tet_with_connectivity query_surface;

    // Build unique points list with their barycentric coordinates
    std::map<uint, int> vertex_map; // Maps arrangement vertex ID to query_surface point index

    for (int i = 0; i < out_tri_ids.size(); i++) {
        int triangle_id = out_tri_ids[i];
        std::cout << "Processing triangle id: " << triangle_id << std::endl;

        uint v0_idx = tm.tri(triangle_id)[0];
        uint v1_idx = tm.tri(triangle_id)[1];
        uint v2_idx = tm.tri(triangle_id)[2];

        // Find containing tetrahedron
        int containing_tet_id = -1;
        for (int tet_id = 0; tet_id < T_out.rows(); tet_id++) {
            if (vertex_to_labels[v0_idx].count(tet_id) &&
                vertex_to_labels[v1_idx].count(tet_id) &&
                vertex_to_labels[v2_idx].count(tet_id)) {
                std::cout << "Triangle " << triangle_id << " is in tet " << tet_id << std::endl;
                containing_tet_id = tet_id;
                break;
            }
        }

        if (containing_tet_id == -1) {
            std::cout << "ERROR! triangle " << triangle_id << " is not in any tet" << std::endl;
            continue;
        }

        // Get tet vertices matrix
        Eigen::Matrix<double, 4, 3> tet_Vs;
        tet_Vs.row(0) = V_out.row(T_out(containing_tet_id, 0));
        tet_Vs.row(1) = V_out.row(T_out(containing_tet_id, 1));
        tet_Vs.row(2) = V_out.row(T_out(containing_tet_id, 2));
        tet_Vs.row(3) = V_out.row(T_out(containing_tet_id, 3));

        // Process each vertex of the triangle
        std::vector<uint> tri_vertex_indices = {v0_idx, v1_idx, v2_idx};
        Eigen::Vector3i query_tri_indices;

        for (int j = 0; j < 3; j++) {
            uint vtx_idx = tri_vertex_indices[j];

            // Check if we've already processed this vertex
            if (vertex_map.find(vtx_idx) == vertex_map.end()) {
                // New vertex - add to points list
                query_point_tet_r qp;
                qp.t_id = containing_tet_id;
                qp.tv_ids = T_out.row(containing_tet_id);

                // Get world position
                Eigen::Vector3d v_world(
                    out_coords[vtx_idx * 3],
                    out_coords[vtx_idx * 3 + 1],
                    out_coords[vtx_idx * 3 + 2]);

                // Compute barycentric coordinates
                Eigen::Vector4d bc_double = world_to_barycentric_tet(v_world, tet_Vs);

                // Convert to rational
                qp.bc = Eigen::Matrix<wmtk::Rational, 4, 1>(
                    wmtk::Rational(bc_double(0)),
                    wmtk::Rational(bc_double(1)),
                    wmtk::Rational(bc_double(2)),
                    wmtk::Rational(bc_double(3))
                );

                // Normalize
                wmtk::Rational sum = qp.bc(0) + qp.bc(1) + qp.bc(2) + qp.bc(3);
                if (sum != wmtk::Rational(0)) {
                    qp.bc = qp.bc / sum;
                }

                // Add to points list
                int point_idx = query_surface.points.size();
                query_surface.points.push_back(qp);
                vertex_map[vtx_idx] = point_idx;
                query_tri_indices(j) = point_idx;
            } else {
                // Reuse existing point
                query_tri_indices(j) = vertex_map[vtx_idx];
            }
        }

        // Add triangle to query_triangles
        query_surface.query_triangles.push_back(query_tri_indices);
        query_surface.tet_ids.push_back(containing_tet_id);
    }

    std::cout << "Created surface with " << query_surface.points.size()
              << " unique points and " << query_surface.query_triangles.size()
              << " triangles" << std::endl;

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

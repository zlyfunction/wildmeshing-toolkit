#include "tet_surface_tracking.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "batch_operation_log_reader.hpp"
#include "tet_point_tracking.hpp"
#include "tet_surface_sampling.hpp"
#include "tet_track_operations.hpp"
#include "tet_track_operations_internal.hpp"
#include "vtu_utils.hpp"

namespace tet_surface_tracking {

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> query_surface_to_world_positions(
    const query_surface_tet& query_surface,
    const Eigen::MatrixXd& V)
{
    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;

    V_out.resize(query_surface.triangles.size() * 3, 3);
    F_out.resize(query_surface.triangles.size(), 3);

    for (size_t i = 0; i < query_surface.triangles.size(); i++) {
        const auto& tri = query_surface.triangles[i];

        // Get vertices of the tetrahedron
        Eigen::Vector4i tet_verts = tri.tv_ids;

        // Calculate real positions for each vertex of the triangle
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3d world_pos = Eigen::Vector3d::Zero();
            for (int k = 0; k < 4; k++) {
                world_pos += tri.bcs[j](k) * V.row(tet_verts(k)).transpose();
            }
            V_out.row(i * 3 + j) = world_pos;
        }

        // Set face indices
        F_out.row(i) = Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2);
    }

    return {V_out, F_out};
}

// helper function to print triangle area statistics
void print_triangle_area_statistics(
    const Eigen::MatrixXd& surface_V,
    const Eigen::MatrixXi& surface_F)
{
    std::cout << "\n=== Triangle Area Statistics ===" << std::endl;

    double total_area = 0.0;
    double min_area = std::numeric_limits<double>::max();
    double max_area = 0.0;
    std::vector<double> areas;

    for (int i = 0; i < surface_F.rows(); i++) {
        Eigen::Vector3d v0 = surface_V.row(surface_F(i, 0));
        Eigen::Vector3d v1 = surface_V.row(surface_F(i, 1));
        Eigen::Vector3d v2 = surface_V.row(surface_F(i, 2));

        Eigen::Vector3d edge1 = v1 - v0;
        Eigen::Vector3d edge2 = v2 - v0;
        Eigen::Vector3d cross_product = edge1.cross(edge2);
        double area = 0.5 * cross_product.norm();

        areas.push_back(area);
        total_area += area;
        min_area = std::min(min_area, area);
        max_area = std::max(max_area, area);
    }

    double mean_area = total_area / areas.size();

    // Calculate standard deviation
    double variance = 0.0;
    for (double area : areas) {
        variance += (area - mean_area) * (area - mean_area);
    }
    variance /= areas.size();
    double std_dev = std::sqrt(variance);

    std::cout << "Number of triangles: " << surface_F.rows() << std::endl;
    std::cout << "Total surface area: " << total_area << std::endl;
    std::cout << "Mean triangle area: " << mean_area << std::endl;
    std::cout << "Min triangle area: " << min_area << std::endl;
    std::cout << "Max triangle area: " << max_area << std::endl;
    std::cout << "Standard deviation: " << std_dev << std::endl;
    std::cout << "Area ratio (max/min): " << (max_area / min_area) << std::endl;
    std::cout << "================================\n" << std::endl;
}
// helper function to check if the surface is manifold
void check_manifold_property(const Eigen::MatrixXd& surface_V, const Eigen::MatrixXi& surface_F)
{
    std::cout << "Merging duplicate vertices..." << std::endl;

    std::vector<Eigen::Vector3d> unique_vertices;
    std::vector<Eigen::Vector3i> unique_faces;

    const double tolerance = 1e-6; // Tolerance for considering vertices as identical

    // Process each triangle
    for (int i = 0; i < surface_F.rows(); i++) {
        Eigen::Vector3i new_face;

        for (int j = 0; j < 3; j++) {
            const Eigen::Vector3d& vertex = surface_V.row(surface_F(i, j));

            // Find if this vertex is close to any existing vertex
            int existing_idx = -1;
            for (size_t k = 0; k < unique_vertices.size(); k++) {
                if ((vertex - unique_vertices[k]).norm() < tolerance) {
                    existing_idx = k;
                    break;
                }
            }

            if (existing_idx == -1) {
                // New vertex
                int new_idx = unique_vertices.size();
                unique_vertices.push_back(vertex);
                new_face(j) = new_idx;
            } else {
                // Existing vertex
                new_face(j) = existing_idx;
            }
        }

        // Only add face if it's not degenerate (all vertices are different)
        if (new_face(0) != new_face(1) && new_face(1) != new_face(2) &&
            new_face(2) != new_face(0)) {
            unique_faces.push_back(new_face);
        }
    }

    // Convert to Eigen matrices
    Eigen::MatrixXd V_triangle(unique_vertices.size(), 3);
    Eigen::MatrixXi F_triangle(unique_faces.size(), 3);

    for (size_t i = 0; i < unique_vertices.size(); i++) {
        V_triangle.row(i) = unique_vertices[i];
    }

    for (size_t i = 0; i < unique_faces.size(); i++) {
        F_triangle.row(i) = unique_faces[i];
    }

    std::cout << "Original mesh: " << surface_V.rows() << " vertices, " << surface_F.rows()
              << " faces" << std::endl;
    std::cout << "Merged mesh: " << V_triangle.rows() << " vertices, " << F_triangle.rows()
              << " faces" << std::endl;

    // Check if the mesh is manifold
    std::cout << "Checking manifold property..." << std::endl;

    // Count edge occurrences and track which faces contain each edge
    std::map<std::pair<int, int>, std::vector<int>> edge_to_faces;
    std::map<std::pair<int, int>, int> edge_count;

    for (int i = 0; i < F_triangle.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            int v1 = F_triangle(i, j);
            int v2 = F_triangle(i, (j + 1) % 3);

            // Ensure consistent edge orientation
            if (v1 > v2) std::swap(v1, v2);

            edge_count[{v1, v2}]++;
            edge_to_faces[{v1, v2}].push_back(i);
        }
    }

    // Check manifold property and collect non-manifold vertices
    bool is_manifold = true;
    int boundary_edges = 0;
    int non_manifold_edges = 0;
    std::set<int> non_manifold_vertices;

    for (const auto& edge : edge_count) {
        if (edge.second == 1) {
            boundary_edges++;
        } else if (edge.second > 2) {
            std::cout << "    Non-manifold edge: (" << edge.first.first << ", " << edge.first.second
                      << ") appears " << edge.second << " times" << std::endl;
            std::cout << "    Edge vertices positions: (" << V_triangle.row(edge.first.first)
                      << ", " << V_triangle.row(edge.first.second) << ")" << std::endl;

            // Print faces containing this non-manifold edge and their areas
            std::cout << "    Faces containing this edge:" << std::endl;
            for (int face_id : edge_to_faces[edge.first]) {
                Eigen::Vector3i face = F_triangle.row(face_id);
                Eigen::Vector3d v0 = V_triangle.row(face(0));
                Eigen::Vector3d v1 = V_triangle.row(face(1));
                Eigen::Vector3d v2 = V_triangle.row(face(2));

                // Calculate triangle area using cross product
                Eigen::Vector3d edge1 = v1 - v0;
                Eigen::Vector3d edge2 = v2 - v0;
                double area = 0.5 * edge1.cross(edge2).norm();

                std::cout << "      Face " << face_id << ": vertices (" << face(0) << ", "
                          << face(1) << ", " << face(2) << "), area = " << area << std::endl;
            }

            non_manifold_edges++;
            is_manifold = false;
            // Add both vertices of non-manifold edges to the set
            non_manifold_vertices.insert(edge.first.first);
            non_manifold_vertices.insert(edge.first.second);
        }
    }

    std::cout << "Manifold check results:" << std::endl;
    std::cout << "  Boundary edges: " << boundary_edges << std::endl;
    std::cout << "  Non-manifold edges: " << non_manifold_edges << std::endl;
    std::cout << "  Is manifold: " << (is_manifold ? "Yes" : "No") << std::endl;

    // Print non-manifold vertices as point mesh
    if (!non_manifold_vertices.empty()) {
        std::cout << "  Non-manifold vertices: " << non_manifold_vertices.size() << std::endl;

        // Create point mesh for non-manifold vertices
        Eigen::MatrixXd V_non_manifold(non_manifold_vertices.size(), 3);
        int idx = 0;
        for (int vertex_id : non_manifold_vertices) {
            V_non_manifold.row(idx) = V_triangle.row(vertex_id);
            idx++;
        }

        // Write non-manifold vertices to file
        vtu_utils::write_point_mesh_to_vtu(V_non_manifold, "non_manifold_vertices.vtu");
        std::cout << "  Non-manifold vertices written to non_manifold_vertices.vtu" << std::endl;
    }
}

void run_back_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold)
{
    std::cout << "Back tracking surface" << std::endl;

    std::string query_surface_filename = surface_file.string();
    query_surface_tet query_surface;

    if (!std::filesystem::exists(query_surface_filename)) {
        std::cout << "query_surface not found, sampling and writing to file..." << std::endl;
        // User must provide surface file or use external sampling functions

        // std::cerr << "Error: Surface file not found. Please provide a valid surface file."
        //   << std::endl;
        query_surface = tet_surface_sampling::sample_query_surface_large_triangle(T_after, V_after);
        write_query_surface_tet_to_file(query_surface, query_surface_filename);
    } else {
        std::cout << "query_surface found, reading from file..." << std::endl;
        query_surface = read_query_surface_tet_from_file(query_surface_filename);
    }

    auto [surface_V, surface_F] = query_surface_to_world_positions(query_surface, V_after);
    vtu_utils::write_triangle_mesh_to_vtu(surface_V, surface_F, "query_surface_tet_after.vtu");

    std::cout << "before tracking, surface size: " << query_surface.triangles.size() << std::endl;
    track_surface_tet(operation_logs_dir, query_surface, false, false);
    std::cout << "after tracking, surface size: " << query_surface.triangles.size() << std::endl;

    write_query_surface_tet_to_file(query_surface, "query_surface_tet_before.json");
    auto [surface_V_before, surface_F_before] =
        query_surface_to_world_positions(query_surface, V_before);
    vtu_utils::write_triangle_mesh_to_vtu(
        surface_V_before,
        surface_F_before,
        "query_surface_tet_before.vtu");

    // Calculate triangle area statistics
    print_triangle_area_statistics(surface_V_before, surface_F_before);

    // Check if the surface is a correct surface
    if (check_manifold) {
        check_manifold_property(surface_V_before, surface_F_before);
    }
}

// handle consolidate surface version
void handle_consolidate_tet_surface(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet& surface,
    bool forward)
{
    std::cout << "Handling Consolidate for surface" << std::endl;
    if (!forward) {
        // backward
        for (auto& qt : surface.triangles) {
            if (qt.t_id >= 0) {
                qt.t_id = tet_ids_maps[qt.t_id];
            }
            for (int j = 0; j < 4; j++) {
                qt.tv_ids[j] = vertex_ids_maps[qt.tv_ids[j]];
            }
        }
    } else {
        // forward
        for (auto& qt : surface.triangles) {
            if (qt.t_id >= 0) {
                auto it = std::find(tet_ids_maps.begin(), tet_ids_maps.end(), qt.t_id);
                if (it != tet_ids_maps.end()) {
                    qt.t_id = std::distance(tet_ids_maps.begin(), it);
                }
                for (int j = 0; j < 4; j++) {
                    auto it_v =
                        std::find(vertex_ids_maps.begin(), vertex_ids_maps.end(), qt.tv_ids[j]);
                    if (it_v != vertex_ids_maps.end()) {
                        qt.tv_ids[j] = std::distance(vertex_ids_maps.begin(), it_v);
                    } else {
                        std::cout << "Error: vertex not found" << std::endl;
                    }
                }
            }
        }
    }
}

/**
 * @brief Helper function to get all possible representations of a vertex
 * @param local_t_id The local tetrahedron ID
 * @param local_bc The barycentric coordinates within the tetrahedron
 * @param T_local The local tetrahedron connectivity matrix
 * @return std::pair<std::vector<int>, std::vector<Eigen::Vector4d>> All possible tetrahedron IDs
 * and their corresponding barycentric coordinates
 */

std::pair<std::vector<int>, std::vector<Eigen::Vector4d>> get_all_possible_representations(
    const int local_t_id,
    const Eigen::Vector4d& local_bc,
    const Eigen::MatrixXi& T_local,
    const double eps = 1e-10)
{
    std::vector<int> all_possible_t_ids;
    std::vector<Eigen::Vector4d> all_possible_bcs;

    // add the original representation
    all_possible_t_ids.push_back(local_t_id);
    all_possible_bcs.push_back(local_bc);

    std::vector<int> non_zeros;
    for (int i = 0; i < 4; i++) {
        if (abs(local_bc(i)) >= eps) {
            non_zeros.push_back(i);
        }
    }

    std::cout << "T_local(local_t_id, non_zeros): ";
    for (int i = 0; i < non_zeros.size(); i++) {
        std::cout << T_local(local_t_id, non_zeros[i]) << " ";
    }
    std::cout << std::endl;

    if (non_zeros.size() < 4) {
        for (int t_id = 0; t_id < T_local.rows(); t_id++) {
            if (t_id == local_t_id) continue;

            bool contains_all = true;
            Eigen::Vector4d bc_tmp = Eigen::Vector4d::Zero();
            for (int i = 0; i < non_zeros.size(); i++) {
                int v_idx = T_local(local_t_id, non_zeros[i]);
                bool found = false;

                for (int j = 0; j < 4; j++) {
                    if (T_local(t_id, j) == v_idx) {
                        found = true;
                        bc_tmp(j) = local_bc(non_zeros[i]);
                        break;
                    }
                }

                if (!found) {
                    contains_all = false;
                    break;
                }
            }

            if (contains_all) {
                all_possible_t_ids.push_back(t_id);
                bc_tmp = bc_tmp / bc_tmp.sum();
                all_possible_bcs.push_back(bc_tmp);
            }
        }
    }

    return {all_possible_t_ids, all_possible_bcs};
}


void handle_local_mapping_tet_surface(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_surface_tet& surface)
{
    // TODO: what can we do with this eps?
    double eps = 1e-10;


    std::cout << "Handling Local Mapping for surface" << std::endl;

    // prepare for the arrangement
    std::vector<double> T_before_coords;
    std::vector<uint> T_before_tris;
    std::vector<uint> T_before_labels;
    for (int i = 0; i < V_before.rows(); i++) {
        for (int j = 0; j < V_before.cols(); j++) {
            T_before_coords.push_back(V_before(i, j));
        }
    }
    for (int t_id = 0; t_id < T_before.rows(); t_id++) {
        auto tet = T_before.row(t_id);

        // Extract the four faces of the tetrahedron
        std::vector<std::vector<int>> faces = {
            {tet[0], tet[1], tet[2]},
            {tet[0], tet[1], tet[3]},
            {tet[0], tet[2], tet[3]},
            {tet[1], tet[2], tet[3]}};

        for (const auto& tri : faces) {
            for (const auto& vertex_id : tri) {
                T_before_tris.push_back(static_cast<uint>(vertex_id));
            }
            // Set label based on tetrahedron ID
            T_before_labels.push_back(static_cast<uint>(t_id));
        }
    }

    int current_surface_triangle_size = surface.triangles.size();
    for (int id = 0; id < current_surface_triangle_size; id++) {
        auto& qt = surface.triangles[id];
        if (qt.t_id >= 0) {
            auto it = std::find(id_map_after.begin(), id_map_after.end(), qt.t_id);
            if (it == id_map_after.end()) continue; // not found in local patch
        }
        query_point_tet qp0 = {qt.t_id, qt.bcs[0], qt.tv_ids};
        query_point_tet qp1 = {qt.t_id, qt.bcs[1], qt.tv_ids};
        query_point_tet qp2 = {qt.t_id, qt.bcs[2], qt.tv_ids};
        std::vector<query_point_tet> qps = {qp0, qp1, qp2};

        // std::cout << "Before local mapping:" << std::endl;
        // for (int i = 0; i < qps.size(); i++) {
        //     std::cout << "Point " << i << ": t_id=" << qps[i].t_id
        //               << ", bc=" << qps[i].bc.transpose() << std::endl;
        // }

        tet_point_tracking::handle_local_mapping_tet(
            V_before,
            T_before,
            id_map_before,
            v_id_map_before,
            V_after,
            T_after,
            id_map_after,
            v_id_map_after,
            qps);

        // Clip barycentric coordinates by epsilon to avoid numerical issues
        // for (auto& qp : qps) {
        //     for (int i = 0; i < 4; i++) {
        //         if (qp.bc[i] < 1e-10) {
        //             qp.bc[i] = 0.0;
        //         } else if (qp.bc[i] > 1.0 - 1e-10) {
        //             qp.bc[i] = 1.0;
        //         }
        //     }
        //     // Renormalize to ensure sum equals 1
        //     double sum = qp.bc.sum();
        //     if (sum > 0) {
        //         qp.bc /= sum;
        //     }
        // }
        std::cout << "After local mapping:" << std::endl;
        for (int i = 0; i < qps.size(); i++) {
            std::cout << "Point " << i << ": t_id=" << qps[i].t_id
                      << ", bc=" << qps[i].bc.transpose() << std::endl;
        }

        std::cout << "Handling one triangle, get intersections." << std::endl;

        // Get local t_ids for each point
        std::vector<int> local_t_ids;
        for (const auto& qp : qps) {
            auto it = std::find(id_map_before.begin(), id_map_before.end(), qp.t_id);
            if (it != id_map_before.end()) {
                int local_id = std::distance(id_map_before.begin(), it);
                local_t_ids.push_back(local_id);
            } else {
                std::cout << "Error: t_id " << qp.t_id << " not found in id_map_before"
                          << std::endl;
                exit(1);
            }
        }
        std::vector<std::vector<int>> all_possible_t_ids_per_point;
        std::vector<std::vector<Eigen::Vector4d>> all_possible_bcs_per_point;

        for (int i = 0; i < qps.size(); i++) {
            auto [t_ids, bcs] =
                get_all_possible_representations(local_t_ids[i], qps[i].bc, T_before, 1e-10);
            all_possible_t_ids_per_point.push_back(t_ids);
            all_possible_bcs_per_point.push_back(bcs);
        }

        // Check if there is a common t_id across all possible representations
        bool found_common_tet = false;
        int common_t_id = -1;

        // Get the first point's possible t_ids as the starting set
        std::set<int> common_t_ids(
            all_possible_t_ids_per_point[0].begin(),
            all_possible_t_ids_per_point[0].end());

        // Intersect with each subsequent point's possible t_ids
        for (int i = 1; i < all_possible_t_ids_per_point.size(); i++) {
            std::set<int> current_t_ids(
                all_possible_t_ids_per_point[i].begin(),
                all_possible_t_ids_per_point[i].end());

            std::set<int> intersection;
            std::set_intersection(
                common_t_ids.begin(),
                common_t_ids.end(),
                current_t_ids.begin(),
                current_t_ids.end(),
                std::inserter(intersection, intersection.begin()));

            common_t_ids = intersection;

            if (common_t_ids.empty()) {
                break;
            }
        }

        if (!common_t_ids.empty()) {
            found_common_tet = true;
            common_t_id = *common_t_ids.begin(); // Take the first common t_id
            std::cout << "Found common tetrahedron: " << common_t_id << std::endl;

            // Set the tetrahedron ID and vertex IDs
            qt.t_id = id_map_before[common_t_id];
            for (int i = 0; i < 4; i++) {
                qt.tv_ids[i] = v_id_map_before[T_before(common_t_id, i)];
            }

            // Get corresponding barycentric coordinates for each point in the common tetrahedron
            for (int i = 0; i < all_possible_t_ids_per_point.size(); i++) {
                for (int j = 0; j < all_possible_t_ids_per_point[i].size(); j++) {
                    if (all_possible_t_ids_per_point[i][j] == common_t_id) {
                        qt.bcs[i] = all_possible_bcs_per_point[i][j];
                        break;
                    }
                }
            }
        } else {
            // std::cout << "No common tetrahedron found across all representations" << std::endl;
            // }
            // // TODO: handle one triangle
            // Check if all 3 points in qps have the same t_id
            // if (qps[0].t_id == qps[1].t_id && qps[1].t_id == qps[2].t_id) {
            //     std::cout << "All points are in one tetrahedron " << qps[0].t_id << std::endl;
            //     // update the query triangle
            //     qt.t_id = qps[0].t_id;
            //     qt.bcs[0] = qps[0].bc;
            //     qt.bcs[1] = qps[1].bc;
            //     qt.bcs[2] = qps[2].bc;
            //     qt.tv_ids = qps[0].tv_ids;
            // }
            // else
            // {
            std::cout << "Not in one tetrahedron" << std::endl;
            std::cout << "Computing arrangement" << std::endl;


            std::vector<double> in_coords = T_before_coords;
            std::vector<uint> in_tris = T_before_tris;
            std::vector<uint> in_labels = T_before_labels;

            // Compute real positions for each query point
            for (int i = 0; i < qps.size(); i++) {
                // Get tetrahedron vertices
                Eigen::Vector4i tet = T_before.row(local_t_ids[i]);
                Eigen::Vector4d bc = qps[i].bc;

                // Compute position using barycentric coordinates
                Eigen::Vector3d pos = Eigen::Vector3d::Zero();
                for (int j = 0; j < 4; j++) {
                    pos += bc[j] * V_before.row(tet[j]).transpose();
                }
                std::cout << "Position " << i << ": (" << pos[0] << ", " << pos[1] << ", " << pos[2]
                          << ")" << std::endl;

                // Add position to in_coords
                in_coords.push_back(pos[0]);
                in_coords.push_back(pos[1]);
                in_coords.push_back(pos[2]);
            }

            // Add triangle to in_tris using base index at end of original vertices
            uint base_idx = V_before.rows();
            in_tris.push_back(base_idx);
            in_tris.push_back(base_idx + 1);
            in_tris.push_back(base_idx + 2);

            // Add label for the triangle
            in_labels.push_back(T_before.rows());


            // init the necessary data structures
            point_arena arena;
            std::vector<genericPoint*> arr_verts;
            std::vector<uint> arr_in_tris, arr_out_tris;
            std::vector<std::bitset<NBIT>> arr_in_labels;
            std::vector<DuplTriInfo> dupl_triangles;
            Labels labels;
            cinolib::Octree octree;
            std::vector<uint> vertex_id_map; // map from original vertices to arranged vertices

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

            // get barycentric coordinates of the output triangles
            std::vector<int> out_tri_ids;
            std::vector<std::set<int>> vertex_to_labels(tm.numVerts());

            for (int i = 0; i < 3; i++) {
                std::cout << "i = " << i << std::endl;

                // Add the primary tetrahedron for this point
                vertex_to_labels[vertex_id_map[V_before.rows() + i]].insert(local_t_ids[i]);

                // Add all possible representations computed earlier
                for (int t_id : all_possible_t_ids_per_point[i]) {
                    vertex_to_labels[vertex_id_map[V_before.rows() + i]].insert(t_id);
                }
            }

            std::cout << "V_before.rows() = " << V_before.rows() << std::endl;
            std::cout << "T_before.rows() = " << T_before.rows() << std::endl;

            for (uint t_id = 0; t_id < tm.numTris(); t_id++) {
                uint v0 = tm.tri(t_id)[0];
                uint v1 = tm.tri(t_id)[1];
                uint v2 = tm.tri(t_id)[2];
                uint v3 = tm.tri(t_id)[3];

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

            {
                std::cout << "\n=== All Vertices and IDs ===" << std::endl;
                for (uint v_id = 0; v_id < tm.numVerts(); v_id++) {
                    if (!vertex_to_labels[v_id].empty()) {
                        std::cout << "Vertex " << v_id << ": ";
                        for (auto label : vertex_to_labels[v_id]) {
                            std::cout << label << " ";
                        }
                        std::cout << "pos: (" << out_coords[3 * v_id] << ", "
                                  << out_coords[3 * v_id + 1] << ", " << out_coords[3 * v_id + 2]
                                  << ")";
                        std::cout << std::endl;
                    }
                }

                for (int i = 0; i < out_tri_ids.size(); i++) {
                    int triangle_id = out_tri_ids[i];
                    uint v0 = tm.tri(triangle_id)[0];
                    uint v1 = tm.tri(triangle_id)[1];
                    uint v2 = tm.tri(triangle_id)[2];
                    std::cout << "Triangle " << triangle_id << ": vertices [" << v0 << ", " << v1
                              << ", " << v2 << "]" << std::endl;
                }
            }

            bool first_triangle = true;
            for (int i = 0; i < out_tri_ids.size(); i++) {
                int triangle_id = out_tri_ids[i];
                std::cout << "checking triangle id: " << triangle_id << std::endl;
                uint v0_idx = tm.tri(triangle_id)[0];
                uint v1_idx = tm.tri(triangle_id)[1];
                uint v2_idx = tm.tri(triangle_id)[2];

                {
                    std::cout << "vertex_to_labels for triangle vertices:" << std::endl;
                    std::cout << "v0 (" << v0_idx << "): ";
                    for (auto label : vertex_to_labels[v0_idx]) {
                        std::cout << label << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "v1 (" << v1_idx << "): ";
                    for (auto label : vertex_to_labels[v1_idx]) {
                        std::cout << label << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "v2 (" << v2_idx << "): ";
                    for (auto label : vertex_to_labels[v2_idx]) {
                        std::cout << label << " ";
                    }
                    std::cout << std::endl;
                }

                int containing_tet_id = -1;
                for (int tet_id = 0; tet_id < T_before.rows(); tet_id++) {
                    if (vertex_to_labels[v0_idx].count(tet_id) &&
                        vertex_to_labels[v1_idx].count(tet_id) &&
                        vertex_to_labels[v2_idx].count(tet_id)) {
                        std::cout << "triangle " << triangle_id << " is in tet " << tet_id
                                  << std::endl;
                        containing_tet_id = tet_id;
                        break;
                    }
                }
                if (containing_tet_id == -1) {
                    std::cout << "Error: triangle " << triangle_id << " is not in any tet"
                              << std::endl;

                    std::ofstream obj_file("debug_triangles.obj");
                    obj_file << "# Debug triangles that couldn't be found in any tet" << std::endl;

                    for (int i = 0; i < out_tri_ids.size(); i++) {
                        int triangle_id = out_tri_ids[i];
                        uint v0_idx = tm.tri(triangle_id)[0];
                        uint v1_idx = tm.tri(triangle_id)[1];
                        uint v2_idx = tm.tri(triangle_id)[2];

                        // Write vertex positions
                        obj_file << "v " << out_coords[v0_idx * 3] << " "
                                 << out_coords[v0_idx * 3 + 1] << " " << out_coords[v0_idx * 3 + 2]
                                 << std::endl;
                        obj_file << "v " << out_coords[v1_idx * 3] << " "
                                 << out_coords[v1_idx * 3 + 1] << " " << out_coords[v1_idx * 3 + 2]
                                 << std::endl;
                        obj_file << "v " << out_coords[v2_idx * 3] << " "
                                 << out_coords[v2_idx * 3 + 1] << " " << out_coords[v2_idx * 3 + 2]
                                 << std::endl;
                    }

                    // Write face indices (1-indexed in OBJ format)
                    for (int i = 0; i < out_tri_ids.size(); i++) {
                        obj_file << "f " << (i * 3 + 1) << " " << (i * 3 + 2) << " " << (i * 3 + 3)
                                 << std::endl;
                    }

                    obj_file.close();
                    std::cout << "Debug triangles written to debug_triangles.obj" << std::endl;
                    continue;
                    // exit(1);
                }

                query_triangle_tet q_tri;
                q_tri.t_id = id_map_before[containing_tet_id];
                Eigen::Matrix<double, 4, 3> tet_Vs;
                tet_Vs.row(0) = V_before.row(T_before(containing_tet_id, 0));
                tet_Vs.row(1) = V_before.row(T_before(containing_tet_id, 1));
                tet_Vs.row(2) = V_before.row(T_before(containing_tet_id, 2));
                tet_Vs.row(3) = V_before.row(T_before(containing_tet_id, 3));
                for (int j = 0; j < 4; j++) {
                    q_tri.tv_ids[j] = v_id_map_before[T_before(containing_tet_id, j)];
                }
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
                q_tri.bcs[0] = world_to_barycentric_tet(v0_world, tet_Vs);
                q_tri.bcs[1] = world_to_barycentric_tet(v1_world, tet_Vs);
                q_tri.bcs[2] = world_to_barycentric_tet(v2_world, tet_Vs);
                q_tri.bcs[0] = q_tri.bcs[0] / q_tri.bcs[0].sum();
                q_tri.bcs[1] = q_tri.bcs[1] / q_tri.bcs[1].sum();
                q_tri.bcs[2] = q_tri.bcs[2] / q_tri.bcs[2].sum();
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 4; k++) {
                        if (std::abs(q_tri.bcs[j](k)) < 1e-15) {
                            q_tri.bcs[j](k) = 0.0;
                        }
                    }
                }
                if (first_triangle) {
                    qt = q_tri;
                    first_triangle = false;
                } else {
                    surface.triangles.push_back(q_tri);
                }
            }
        }
    }
}

void write_query_surface_tet_to_file(const query_surface_tet& surface, const std::string& filename)
{
    json j;
    j["num_triangles"] = surface.triangles.size();

    for (size_t i = 0; i < surface.triangles.size(); ++i) {
        const auto& tri = surface.triangles[i];
        json tri_json;
        tri_json["t_id"] = tri.t_id;

        // Store barycentric coordinates
        for (int j = 0; j < 3; ++j) {
            tri_json["bcs"][j] = {tri.bcs[j][0], tri.bcs[j][1], tri.bcs[j][2], tri.bcs[j][3]};
        }

        // Store tetrahedron vertex ids
        tri_json["tv_ids"] = {tri.tv_ids[0], tri.tv_ids[1], tri.tv_ids[2], tri.tv_ids[3]};

        j["triangles"].push_back(tri_json);
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
    } else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

query_surface_tet read_query_surface_tet_from_file(const std::string& filename)
{
    query_surface_tet surface;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return surface;
    }

    json j;
    file >> j;
    file.close();

    size_t num_triangles = j["num_triangles"];
    surface.triangles.resize(num_triangles);

    for (size_t i = 0; i < num_triangles; ++i) {
        const auto& tri_json = j["triangles"][i];
        auto& tri = surface.triangles[i];

        tri.t_id = tri_json["t_id"];

        // Read barycentric coordinates
        for (int j = 0; j < 3; ++j) {
            const auto& bc_array = tri_json["bcs"][j];
            tri.bcs[j] = Eigen::Vector4d(bc_array[0], bc_array[1], bc_array[2], bc_array[3]);
        }

        // Read tetrahedron vertex ids
        const auto& tv_array = tri_json["tv_ids"];
        tri.tv_ids = Eigen::Vector4i(tv_array[0], tv_array[1], tv_array[2], tv_array[3]);
    }

    return surface;
}

void track_surface_one_operation_tet(
    const json& operation_log,
    query_surface_tet& query_surface,
    bool do_forward,
    bool use_rational,
    int operation_id)
{
    std::string operation_name = operation_log["operation_name"];
    if (operation_name == "MeshConsolidate") {
        std::cout << "This Operations is Consolidate" << std::endl;
        std::vector<int64_t> tet_ids_maps;
        std::vector<int64_t> vertex_ids_maps;
        parse_consolidate_file_tet(operation_log, tet_ids_maps, vertex_ids_maps);

        handle_consolidate_tet_surface(tet_ids_maps, vertex_ids_maps, query_surface, do_forward);
    } else {
        std::cout << "This Operations is " << operation_name << std::endl;
        Eigen::MatrixXi T_after, T_before;
        Eigen::MatrixXd V_after, V_before;
        std::vector<int64_t> id_map_after, id_map_before;
        std::vector<int64_t> v_id_map_after, v_id_map_before;
        parse_non_collapse_file_tet(
            operation_log,
            V_before,
            T_before,
            id_map_before,
            v_id_map_before,
            V_after,
            T_after,
            id_map_after,
            v_id_map_after,
            operation_id);

        if (do_forward) {
            handle_local_mapping_tet_surface(
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                query_surface);
        } else {
            handle_local_mapping_tet_surface(
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                query_surface);
        }
    }
}

void track_surface_tet(
    const std::filesystem::path& dirPath,
    query_surface_tet& query_surface,
    bool do_forward,
    bool use_rational)
{
    BatchOperationLogReader reader(dirPath);
    size_t total_ops = reader.get_total_operations();

    if (total_ops == 0) {
        std::cerr << "No operation logs found in " << dirPath << std::endl;
        return;
    }

    std::cout << "Found " << total_ops << " operations in "
              << (reader.is_batch_format() ? "batch" : "legacy") << " format" << std::endl;

    for (size_t i = 0; i < total_ops; ++i) {
        size_t operation_index = i;
        if (!do_forward) {
            operation_index = total_ops - 1 - i;
        }

        json operation_log = reader.get_operation(operation_index);
        if (operation_log.empty()) {
            std::cerr << "Failed to read operation " << operation_index << std::endl;
            continue;
        }

        std::cout << "Trace Operations number: " << operation_index << std::endl;
        track_surface_one_operation_tet(
            operation_log,
            query_surface,
            do_forward,
            use_rational,
            static_cast<int>(operation_index));
    }
}

} // namespace tet_surface_tracking

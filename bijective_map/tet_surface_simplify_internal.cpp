#include "tet_surface_simplify_internal.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include "tet_track_operations.hpp"
#include "vtu_utils.hpp"

namespace tet_surface_tracking_with_connectivity {

namespace {
// Helper function to write triangle mesh to VTU with point data (global vertex IDs)
void write_triangle_mesh_to_vtu_with_point_data(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXi& point_global_ids,
    const std::string& filename)
{
    Eigen::MatrixXd V3;
    if (V.cols() == 3) {
        V3 = V;
    } else if (V.cols() == 2) {
        V3.resize(V.rows(), 3);
        V3.leftCols(2) = V;
        V3.col(2).setZero();
    } else {
        std::cerr
            << "write_triangle_mesh_to_vtu_with_point_data expects V with 2 or 3 columns, got "
            << V.cols() << std::endl;
        return;
    }
    if (point_global_ids.size() != V3.rows()) {
        std::cerr << "write_triangle_mesh_to_vtu_with_point_data: point_global_ids size ("
                  << point_global_ids.size() << ") does not match number of vertices (" << V3.rows()
                  << ")" << std::endl;
        return;
    }
    if (V3.rows() == 0 || F.rows() == 0) {
        std::cerr << "write_triangle_mesh_to_vtu_with_point_data: empty mesh, skipping write"
                  << std::endl;
        return;
    }
    // Validate F indices
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            if (F(i, j) < 0 || F(i, j) >= V3.rows()) {
                std::cerr << "write_triangle_mesh_to_vtu_with_point_data: invalid face index F("
                          << i << "," << j << ")=" << F(i, j) << " (max=" << V3.rows() - 1 << ")"
                          << std::endl;
                return;
            }
        }
    }
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "write_triangle_mesh_to_vtu_with_point_data: failed to open file " << filename
                  << std::endl;
        return;
    }
    outfile.precision(15);
    outfile << "<?xml version=\"1.0\"?>\n";
    outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    outfile << "  <UnstructuredGrid>\n";
    outfile << "    <Piece NumberOfPoints=\"" << V3.rows() << "\" NumberOfCells=\"" << F.rows()
            << "\">\n";
    outfile << "      <Points>\n";
    outfile << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    int point_count = 0;
    for (int i = 0; i < V3.rows(); i++) {
        double x = V3(i, 0);
        double y = V3(i, 1);
        double z = V3(i, 2);
        if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isinf(x) || std::isinf(y) ||
            std::isinf(z)) {
            std::cerr << "write_triangle_mesh_to_vtu_with_point_data: invalid point at index " << i
                      << ": (" << x << "," << y << "," << z << ")" << std::endl;
            outfile.close();
            return;
        }
        outfile << "          " << x << " " << y << " " << z << "\n";
        point_count++;
    }
    if (point_count != V3.rows()) {
        std::cerr << "write_triangle_mesh_to_vtu_with_point_data: point count mismatch: wrote "
                  << point_count << " but expected " << V3.rows() << std::endl;
        outfile.close();
        return;
    }
    outfile << "        </DataArray>\n";
    outfile << "      </Points>\n";
    outfile << "      <PointData>\n";
    outfile << "        <DataArray type=\"Int64\" Name=\"global_vertex_id\" format=\"ascii\">\n";
    for (int i = 0; i < point_global_ids.size(); ++i) {
        outfile << "          " << point_global_ids(i) << "\n";
    }
    outfile << "        </DataArray>\n";
    outfile << "      </PointData>\n";
    outfile << "      <Cells>\n";
    outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < F.rows(); i++) {
        outfile << "          " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2) << "\n";
    }
    outfile << "        </DataArray>\n";
    outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 0; i < F.rows(); i++) {
        outfile << "          " << (i + 1) * 3 << "\n";
    }
    outfile << "        </DataArray>\n";
    outfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < F.rows(); i++) {
        outfile << "          5\n";
    }
    outfile << "        </DataArray>\n";
    outfile << "      </Cells>\n";
    outfile << "    </Piece>\n";
    outfile << "  </UnstructuredGrid>\n";
    outfile << "</VTKFile>\n";
    outfile.close();
}
// Helper function to build loops from edges
// Takes a list of edges and assembles them into one or more closed loops
void build_loops_from_edges(
    const std::vector<std::pair<int, int>>& edges,
    std::vector<std::vector<int>>& loops)
{
    loops.clear();
    if (edges.empty()) {
        return;
    }
    std::map<int, std::vector<int>> vertex_neighbors;
    for (const auto& edge : edges) {
        if (edge.first != edge.second) {
            vertex_neighbors[edge.first].push_back(edge.second);
            vertex_neighbors[edge.second].push_back(edge.first);
        }
    }
    std::set<std::pair<int, int>> used_edges;
    for (const auto& edge : edges) {
        if (edge.first == edge.second) {
            continue;
        }
        std::pair<int, int> normalized_edge = (edge.first < edge.second)
                                                  ? std::make_pair(edge.first, edge.second)
                                                  : std::make_pair(edge.second, edge.first);
        if (used_edges.find(normalized_edge) != used_edges.end()) {
            continue;
        }
        std::vector<int> loop;
        int start_vertex = edge.first;
        int current_vertex = start_vertex;
        int prev_vertex = -1;
        bool found_loop = false;
        int max_iterations = static_cast<int>(edges.size() * 2);
        int iterations = 0;
        while (iterations++ < max_iterations) {
            loop.push_back(current_vertex);
            if (current_vertex == start_vertex && loop.size() > 2) {
                found_loop = true;
                break;
            }
            bool found_next = false;
            for (int next : vertex_neighbors[current_vertex]) {
                if (next == prev_vertex) {
                    continue;
                }
                std::pair<int, int> edge_to_check = (current_vertex < next)
                                                        ? std::make_pair(current_vertex, next)
                                                        : std::make_pair(next, current_vertex);
                if (used_edges.find(edge_to_check) == used_edges.end()) {
                    used_edges.insert(edge_to_check);
                    prev_vertex = current_vertex;
                    current_vertex = next;
                    found_next = true;
                    break;
                }
            }
            if (!found_next) {
                break;
            }
        }
        if (found_loop && loop.size() >= 3) {
            loops.push_back(loop);
        }
    }
}
// Helper function to perform ear clipping triangulation on a polygon
// Input: boundary_loop - ordered list of vertex indices forming a closed loop
//        points - vector of points (used for validation, not required for simple ear clipping)
// Output: triangles - vector of triangles (each triangle is Vector3i)
// Note: This is a simplified ear clipping that works for simple polygons
void ear_clipping_triangulation(
    const std::vector<int>& boundary_loop,
    const std::vector<query_point_tet_r>& /*points*/,
    std::vector<Eigen::Vector3i>& triangles)
{
    std::cout << "      Ear clipping input polygon: [";
    for (size_t i = 0; i < boundary_loop.size(); ++i) {
        std::cout << boundary_loop[i];
        if (i < boundary_loop.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "] (" << boundary_loop.size() << " vertices)" << std::endl;
    if (boundary_loop.size() < 3) {
        std::cout << "      Input polygon too small, skipping" << std::endl;
        return;
    }
    if (boundary_loop.size() == 3) {
        if (boundary_loop[0] != boundary_loop[1] && boundary_loop[1] != boundary_loop[2] &&
            boundary_loop[0] != boundary_loop[2]) {
            triangles.push_back(
                Eigen::Vector3i(boundary_loop[0], boundary_loop[1], boundary_loop[2]));
            std::cout << "      Single triangle: [" << boundary_loop[0] << "," << boundary_loop[1]
                      << "," << boundary_loop[2] << "]" << std::endl;
        }
        return;
    }
    std::vector<int> poly = boundary_loop;
    for (auto it = poly.begin(); it != poly.end();) {
        bool has_duplicate = false;
        for (auto it2 = poly.begin(); it2 != poly.end(); ++it2) {
            if (it2 != it && *it2 == *it) {
                has_duplicate = true;
                break;
            }
        }
        if (has_duplicate) {
            it = poly.erase(it);
        } else {
            ++it;
        }
    }
    if (poly.size() < 3) {
        std::cout << "      Warning: Polygon has less than 3 vertices after removing duplicates"
                  << std::endl;
        return;
    }
    int max_iterations = static_cast<int>(poly.size() * poly.size());
    int iterations = 0;
    while (poly.size() > 3 && iterations < max_iterations) {
        iterations++;
        bool ear_found = false;
        for (size_t i = 0; i < poly.size(); ++i) {
            int prev_idx = (i == 0) ? static_cast<int>(poly.size()) - 1 : static_cast<int>(i) - 1;
            int curr_idx = static_cast<int>(i);
            int next_idx = (i == poly.size() - 1) ? 0 : static_cast<int>(i) + 1;
            int v0 = poly[prev_idx];
            int v1 = poly[curr_idx];
            int v2 = poly[next_idx];
            if (v0 == v1 || v1 == v2 || v0 == v2) {
                continue;
            }
            triangles.push_back(Eigen::Vector3i(v0, v1, v2));
            poly.erase(poly.begin() + curr_idx);
            ear_found = true;
            break;
        }
        if (!ear_found) {
            break;
        }
    }
    if (poly.size() == 3) {
        if (poly[0] != poly[1] && poly[1] != poly[2] && poly[0] != poly[2]) {
            triangles.push_back(Eigen::Vector3i(poly[0], poly[1], poly[2]));
        }
    } else if (poly.size() > 3) {
        std::cout
            << "      Warning: Ear clipping incomplete, using fan triangulation for remaining "
            << poly.size() << " vertices" << std::endl;
        for (size_t i = 1; i < poly.size() - 1; ++i) {
            int v0 = poly[0];
            int v1 = poly[i];
            int v2 = poly[i + 1];
            if (v0 != v1 && v1 != v2 && v0 != v2) {
                triangles.push_back(Eigen::Vector3i(v0, v1, v2));
            }
        }
    }
}
} // namespace

// Simplify refined triangles by grouping them by tet_id and removing interior points
// For each tet_id's subsurface, if it has interior points, extract boundary loop
// and retriangulate using ear clipping
// This function processes triangles starting from start_tri_idx
void simplify_refined_triangles_by_tet(
    query_surface_tet_with_connectivity& surface,
    size_t start_tri_idx,
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    int operation_id)
{
    std::cout << "\n=== Starting simplification of refined triangles ===" << std::endl;
    std::cout << "  Processing triangles from index " << start_tri_idx << " to "
              << surface.query_triangles.size() << std::endl;
    if (start_tri_idx >= surface.query_triangles.size()) {
        std::cout << "  No refined triangles to simplify" << std::endl;
        return;
    }
    size_t num_refined_triangles = surface.query_triangles.size() - start_tri_idx;
    std::cout << "  Found " << num_refined_triangles << " refined triangles to process"
              << std::endl;
    std::map<int64_t, std::vector<size_t>> tet_id_to_triangles;
    for (size_t i = start_tri_idx; i < surface.query_triangles.size(); ++i) {
        if (i < surface.tet_ids.size()) {
            int64_t tet_id = surface.tet_ids[i];
            tet_id_to_triangles[tet_id].push_back(i);
        }
    }
    std::cout << "  Grouped refined triangles into " << tet_id_to_triangles.size() << " tet_ids"
              << std::endl;
    std::vector<bool> triangle_to_remove(surface.query_triangles.size(), false);
    std::set<int> points_to_remove;
    std::vector<Eigen::Vector3i> new_triangles;
    std::vector<int> new_tet_ids;

    for (const auto& [tet_id, tri_indices] : tet_id_to_triangles) {
        std::cout << "\n  Processing tet_id " << tet_id << " with " << tri_indices.size()
                  << " refined triangles" << std::endl;


        // Save triangles before simplification
        std::set<int> local_points_before;
        for (size_t tri_idx : tri_indices) {
            const auto& tri = surface.query_triangles[tri_idx];
            local_points_before.insert(tri(0));
            local_points_before.insert(tri(1));
            local_points_before.insert(tri(2));
        }
        Eigen::MatrixXd V_before_simplify(local_points_before.size(), 3);
        Eigen::MatrixXi F_before_simplify(tri_indices.size(), 3);
        Eigen::VectorXi point_global_ids_before(local_points_before.size());
        // Map from original point index (in surface.points) to local index (0, 1, 2, ...) in
        // V_before_simplify
        std::map<int, int> original_to_local_before;
        int local_idx = 0;
        for (int p_idx : local_points_before) {
            original_to_local_before[p_idx] = local_idx;
            const auto& qp = surface.points[p_idx];
            Eigen::Matrix<double, 4, 3> tet_vertices;
            Eigen::Vector4d bc_double;
            for (int i = 0; i < 4; ++i) {
                bc_double(i) = qp.bc(i).to_double();
            }
            // Map global tet ID to local index in T_before
            int local_tet_idx = -1;
            if (qp.t_id >= 0) {
                auto it = std::find(id_map_before.begin(), id_map_before.end(), qp.t_id);
                if (it != id_map_before.end()) {
                    local_tet_idx = std::distance(id_map_before.begin(), it);
                }
            }
            if (local_tet_idx >= 0 && local_tet_idx < T_before.rows()) {
                // Use tet ID to get vertices
                for (int i = 0; i < 4; ++i) {
                    int v_id = T_before(local_tet_idx, i);
                    if (v_id >= 0 && v_id < V_before.rows()) {
                        tet_vertices.row(i) = Eigen::Vector3d(
                            V_before(v_id, 0).to_double(),
                            V_before(v_id, 1).to_double(),
                            V_before(v_id, 2).to_double());
                    }
                }
            } else {
                // Fallback: use tv_ids and v_id_map_before to get vertex coordinates
                if (qp.tv_ids.size() != 4) {
                    throw std::runtime_error(
                        "Error: simplify_refined_triangles_by_tet: qp.tv_ids.size() != 4");
                }
                for (int i = 0; i < 4; ++i) {
                    if (bc_double(i) != 0.0) {
                        int64_t global_v_id = qp.tv_ids(i);
                        auto it =
                            std::find(v_id_map_before.begin(), v_id_map_before.end(), global_v_id);
                        if (it == v_id_map_before.end()) {
                            throw std::runtime_error(
                                "Error: simplify_refined_triangles_by_tet: vertex ID " +
                                std::to_string(global_v_id) + " not found in v_id_map_before (bc[" +
                                std::to_string(i) + "] != 0)");
                        }
                        int local_v_idx = std::distance(v_id_map_before.begin(), it);
                        if (local_v_idx >= 0 && local_v_idx < V_before.rows()) {
                            tet_vertices.row(i) = Eigen::Vector3d(
                                V_before(local_v_idx, 0).to_double(),
                                V_before(local_v_idx, 1).to_double(),
                                V_before(local_v_idx, 2).to_double());
                        }
                    }
                }
            }
            Eigen::Vector3d world_pos = barycentric_to_world_tet<double>(bc_double, tet_vertices);

            V_before_simplify.row(local_idx) = world_pos.transpose();

            point_global_ids_before(local_idx) = p_idx;
            local_idx++;
        }
        // Build F using local indices (0, 1, 2, ...) corresponding to rows in V_before_simplify
        for (size_t i = 0; i < tri_indices.size(); ++i) {
            size_t tri_idx = tri_indices[i];
            const auto& tri = surface.query_triangles[tri_idx];
            F_before_simplify.row(i) = Eigen::Vector3i(
                original_to_local_before[tri(0)],
                original_to_local_before[tri(1)],
                original_to_local_before[tri(2)]);
        }
        std::string filename_before = "simplify_before_op" + std::to_string(operation_id) + "_tet" +
                                      std::to_string(tet_id) + ".vtu";
        std::cout << "    Writing before simplification: V.rows()=" << V_before_simplify.rows()
                  << ", F.rows()=" << F_before_simplify.rows()
                  << ", point_global_ids.size()=" << point_global_ids_before.size() << std::endl;
        write_triangle_mesh_to_vtu_with_point_data(
            V_before_simplify,
            F_before_simplify,
            point_global_ids_before,
            filename_before);
        std::cout << "    Saved before simplification to: " << filename_before << std::endl;


        std::set<int> local_points;
        for (size_t tri_idx : tri_indices) {
            const auto& tri = surface.query_triangles[tri_idx];
            local_points.insert(tri(0));
            local_points.insert(tri(1));
            local_points.insert(tri(2));
        }
        std::map<std::pair<int, int>, int> edge_count;
        for (size_t tri_idx : tri_indices) {
            const auto& tri = surface.query_triangles[tri_idx];
            for (int j = 0; j < 3; ++j) {
                int v0 = tri(j);
                int v1 = tri((j + 1) % 3);
                if (v0 > v1) std::swap(v0, v1);
                edge_count[{v0, v1}]++;
            }
        }
        std::set<int> boundary_points;
        std::vector<std::pair<int, int>> boundary_edges;
        for (const auto& [edge, count] : edge_count) {
            if (count == 1) {
                boundary_edges.push_back(edge);
                boundary_points.insert(edge.first);
                boundary_points.insert(edge.second);
            }
        }
        std::set<int> interior_points;
        for (int p : local_points) {
            if (boundary_points.find(p) == boundary_points.end()) {
                interior_points.insert(p);
            }
        }
        std::cout << "    Boundary points: " << boundary_points.size()
                  << ", Interior points: " << interior_points.size() << std::endl;
        if (interior_points.empty()) {
            std::cout << "    No interior points, skipping tet_id " << tet_id << std::endl;
            continue;
        }
        if (boundary_edges.empty()) {
            std::cout << "    No boundary edges found, skipping" << std::endl;
            continue;
        }
        std::cout << "    Found interior points, processing connected components using BFS..."
                  << std::endl;
        // Build point-to-triangles map
        std::map<int, std::vector<size_t>> point_to_triangles;
        for (size_t tri_idx : tri_indices) {
            const auto& tri = surface.query_triangles[tri_idx];
            for (int j = 0; j < 3; ++j) {
                point_to_triangles[tri(j)].push_back(tri_idx);
            }
        }
        // Build adjacency map between interior points (through triangles)
        std::map<int, std::set<int>> interior_adjacency;
        for (size_t tri_idx : tri_indices) {
            const auto& tri = surface.query_triangles[tri_idx];
            std::vector<int> tri_interior_points;
            for (int j = 0; j < 3; ++j) {
                int v = tri(j);
                if (interior_points.find(v) != interior_points.end()) {
                    tri_interior_points.push_back(v);
                }
            }
            // Connect all interior points in this triangle
            for (size_t i = 0; i < tri_interior_points.size(); ++i) {
                for (size_t j = i + 1; j < tri_interior_points.size(); ++j) {
                    interior_adjacency[tri_interior_points[i]].insert(tri_interior_points[j]);
                    interior_adjacency[tri_interior_points[j]].insert(tri_interior_points[i]);
                }
            }
        }
        // Process connected components of interior points using BFS
        std::vector<std::vector<int>> all_loops_to_triangulate;
        std::set<size_t> triangles_to_remove_set;
        std::set<int> visited_interior;
        for (int start_interior : interior_points) {
            if (visited_interior.find(start_interior) != visited_interior.end()) {
                continue;
            }
            std::cout << "    Processing connected component starting from interior point "
                      << start_interior << std::endl;
            // BFS to collect all connected interior points
            std::queue<int> bfs_queue;
            std::set<int> component_interior_points;
            bfs_queue.push(start_interior);
            visited_interior.insert(start_interior);
            component_interior_points.insert(start_interior);
            while (!bfs_queue.empty()) {
                int current_interior = bfs_queue.front();
                bfs_queue.pop();
                auto adj_it = interior_adjacency.find(current_interior);
                if (adj_it != interior_adjacency.end()) {
                    for (int neighbor : adj_it->second) {
                        if (visited_interior.find(neighbor) == visited_interior.end()) {
                            visited_interior.insert(neighbor);
                            component_interior_points.insert(neighbor);
                            bfs_queue.push(neighbor);
                        }
                    }
                }
            }
            std::cout << "      Component contains " << component_interior_points.size()
                      << " interior point(s): [";
            bool first = true;
            for (int p : component_interior_points) {
                if (!first) {
                    std::cout << ", ";
                }
                std::cout << p;
                first = false;
            }
            std::cout << "]" << std::endl;
            // Collect all boundary edges from triangles incident to this component
            std::set<std::pair<int, int>> component_boundary_edges;
            std::set<size_t> component_triangles;
            for (int interior_point : component_interior_points) {
                auto it = point_to_triangles.find(interior_point);
                if (it == point_to_triangles.end()) {
                    throw std::runtime_error(
                        "Error: simplify_refined_triangles_by_tet: interior point " +
                        std::to_string(interior_point) + " not found in point_to_triangles");
                }
                const auto& incident_tris = it->second;
                for (size_t tri_idx : incident_tris) {
                    component_triangles.insert(tri_idx);
                    const auto& tri = surface.query_triangles[tri_idx];
                    std::vector<int> boundary_vertices;
                    for (int j = 0; j < 3; ++j) {
                        int v = tri(j);
                        if (component_interior_points.find(v) != component_interior_points.end()) {
                            continue;
                        }
                        if (boundary_points.find(v) != boundary_points.end()) {
                            boundary_vertices.push_back(v);
                        }
                    }
                    // Add edges between boundary vertices in this triangle
                    if (boundary_vertices.size() == 2) {
                        std::pair<int, int> edge =
                            (boundary_vertices[0] < boundary_vertices[1])
                                ? std::make_pair(boundary_vertices[0], boundary_vertices[1])
                                : std::make_pair(boundary_vertices[1], boundary_vertices[0]);
                        component_boundary_edges.insert(edge);
                    } else if (boundary_vertices.size() == 3) {
                        for (int i = 0; i < 3; ++i) {
                            int v0 = boundary_vertices[i];
                            int v1 = boundary_vertices[(i + 1) % 3];
                            std::pair<int, int> edge =
                                (v0 < v1) ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
                            component_boundary_edges.insert(edge);
                        }
                    }
                }
            }
            // Convert set of edges to vector for build_loops_from_edges
            std::vector<std::pair<int, int>> component_edges_vec(
                component_boundary_edges.begin(),
                component_boundary_edges.end());
            // Build loops from these boundary edges
            std::vector<std::vector<int>> component_loops;
            build_loops_from_edges(component_edges_vec, component_loops);
            std::cout << "      Built " << component_loops.size() << " loop(s) for this component"
                      << std::endl;
            for (auto& loop : component_loops) {
                if (loop.size() < 4) {
                    std::cout << "        Skipping small loop (" << loop.size() << " vertices)"
                              << std::endl;
                    continue;
                }
                // Remove duplicate start vertex if present
                if (!loop.empty() && loop.front() == loop.back()) {
                    loop.pop_back();
                }
                if (loop.size() < 3) {
                    continue;
                }
                std::cout << "        Loop: [";
                for (size_t i = 0; i < loop.size(); ++i) {
                    std::cout << loop[i];
                    if (i < loop.size() - 1) {
                        std::cout << ",";
                    }
                }
                std::cout << "] (" << loop.size() << " vertices)" << std::endl;
                all_loops_to_triangulate.push_back(loop);
            }
            // Mark triangles incident to this component for removal
            for (size_t tri_idx : component_triangles) {
                triangles_to_remove_set.insert(tri_idx);
            }
        }
        if (all_loops_to_triangulate.empty()) {
            std::cout << "    No loops available for ear clipping, skipping tet_id " << tet_id
                      << std::endl;
            continue;
        }
        // Perform ear clipping for each loop
        std::vector<Eigen::Vector3i> all_triangulated_triangles;
        size_t loop_counter = 0;
        for (const auto& loop : all_loops_to_triangulate) {
            std::cout << "    Ear clipping loop " << loop_counter++ << " with " << loop.size()
                      << " vertices" << std::endl;
            std::vector<Eigen::Vector3i> triangulated_triangles;
            ear_clipping_triangulation(loop, surface.points, triangulated_triangles);
            std::cout << "      Generated " << triangulated_triangles.size() << " triangles"
                      << std::endl;
            for (size_t i = 0; i < triangulated_triangles.size(); ++i) {
                const auto& tri = triangulated_triangles[i];
                std::cout << "        Triangle " << i << ": [" << tri(0) << "," << tri(1) << ","
                          << tri(2) << "]" << std::endl;
                all_triangulated_triangles.push_back(tri);
            }
        }
        // Save triangles after simplification
        // Collect all triangles that will remain: new triangulated triangles + triangles not
        // removed
        std::vector<Eigen::Vector3i> all_final_triangles;
        all_final_triangles.reserve(all_triangulated_triangles.size() + tri_indices.size());
        // Add new triangulated triangles from ear clipping
        for (const auto& tri : all_triangulated_triangles) {
            all_final_triangles.push_back(tri);
        }
        // Add triangles that are not removed (for this tet_id)
        // These are triangles that belong to this tet_id but are not marked for removal
        for (size_t tri_idx : tri_indices) {
            if (triangles_to_remove_set.find(tri_idx) == triangles_to_remove_set.end()) {
                all_final_triangles.push_back(surface.query_triangles[tri_idx]);
            }
        }
        std::cout << "    After simplification: " << all_triangulated_triangles.size()
                  << " new triangles from ear clipping, "
                  << (all_final_triangles.size() - all_triangulated_triangles.size())
                  << " preserved triangles, total: " << all_final_triangles.size() << " triangles"
                  << std::endl;
        // Collect all points used by final triangles
        std::set<int> local_points_after;
        for (const auto& tri : all_final_triangles) {
            local_points_after.insert(tri(0));
            local_points_after.insert(tri(1));
            local_points_after.insert(tri(2));
        }
        Eigen::MatrixXd V_after_simplify(local_points_after.size(), 3);
        Eigen::MatrixXi F_after_simplify(all_final_triangles.size(), 3);
        Eigen::VectorXi point_global_ids_after(local_points_after.size());
        // Map from original point index (in surface.points) to local index (0, 1, 2, ...) in
        // V_after_simplify
        std::map<int, int> original_to_local_after;
        int local_idx_after = 0;
        for (int p_idx : local_points_after) {
            original_to_local_after[p_idx] = local_idx_after;
            const auto& qp = surface.points[p_idx];
            Eigen::Matrix<double, 4, 3> tet_vertices;
            Eigen::Vector4d bc_double;
            for (int i = 0; i < 4; ++i) {
                bc_double(i) = qp.bc(i).to_double();
            }
            // Map global tet ID to local index in T_before
            int local_tet_idx = -1;
            if (qp.t_id >= 0) {
                auto it = std::find(id_map_before.begin(), id_map_before.end(), qp.t_id);
                if (it != id_map_before.end()) {
                    local_tet_idx = std::distance(id_map_before.begin(), it);
                }
            }
            if (local_tet_idx >= 0 && local_tet_idx < T_before.rows()) {
                // Use tet ID to get vertices
                for (int i = 0; i < 4; ++i) {
                    int v_id = T_before(local_tet_idx, i);
                    if (v_id >= 0 && v_id < V_before.rows()) {
                        tet_vertices.row(i) = Eigen::Vector3d(
                            V_before(v_id, 0).to_double(),
                            V_before(v_id, 1).to_double(),
                            V_before(v_id, 2).to_double());
                    }
                }
            } else {
                // Fallback: use tv_ids and v_id_map_before to get vertex coordinates
                if (qp.tv_ids.size() != 4) {
                    throw std::runtime_error(
                        "Error: simplify_refined_triangles_by_tet: qp.tv_ids.size() != 4");
                }
                for (int i = 0; i < 4; ++i) {
                    if (bc_double(i) != 0.0) {
                        int64_t global_v_id = qp.tv_ids(i);
                        auto it =
                            std::find(v_id_map_before.begin(), v_id_map_before.end(), global_v_id);
                        if (it == v_id_map_before.end()) {
                            throw std::runtime_error(
                                "Error: simplify_refined_triangles_by_tet: vertex ID " +
                                std::to_string(global_v_id) + " not found in v_id_map_before (bc[" +
                                std::to_string(i) + "] != 0)");
                        }
                        int local_v_idx = std::distance(v_id_map_before.begin(), it);
                        if (local_v_idx >= 0 && local_v_idx < V_before.rows()) {
                            tet_vertices.row(i) = Eigen::Vector3d(
                                V_before(local_v_idx, 0).to_double(),
                                V_before(local_v_idx, 1).to_double(),
                                V_before(local_v_idx, 2).to_double());
                        }
                    }
                }
            }
            Eigen::Vector3d world_pos = barycentric_to_world_tet<double>(bc_double, tet_vertices);
            V_after_simplify.row(local_idx_after) = world_pos.transpose();

            point_global_ids_after(local_idx_after) = p_idx;
            local_idx_after++;
        }
        // Build F using local indices (0, 1, 2, ...) corresponding to rows in V_after_simplify
        for (size_t i = 0; i < all_final_triangles.size(); ++i) {
            const auto& tri = all_final_triangles[i];
            F_after_simplify.row(i) = Eigen::Vector3i(
                original_to_local_after[tri(0)],
                original_to_local_after[tri(1)],
                original_to_local_after[tri(2)]);
        }
        std::string filename_after = "simplify_after_op" + std::to_string(operation_id) + "_tet" +
                                     std::to_string(tet_id) + ".vtu";
        std::cout << "    Writing after simplification: V.rows()=" << V_after_simplify.rows()
                  << ", F.rows()=" << F_after_simplify.rows()
                  << ", point_global_ids.size()=" << point_global_ids_after.size() << std::endl;
        write_triangle_mesh_to_vtu_with_point_data(
            V_after_simplify,
            F_after_simplify,
            point_global_ids_after,
            filename_after);
        std::cout << "    Saved after simplification to: " << filename_after << std::endl;
        for (const auto& tri : all_triangulated_triangles) {
            new_triangles.push_back(tri);
            new_tet_ids.push_back(static_cast<int>(tet_id));
        }
        // Mark triangles for removal
        for (size_t tri_idx : triangles_to_remove_set) {
            triangle_to_remove[tri_idx] = true;
        }
        for (int p : interior_points) {
            points_to_remove.insert(p);
        }
    }
    std::cout << "\n  Removing " << points_to_remove.size() << " interior points" << std::endl;
    std::cout << "  Removed point IDs: [";
    bool first_point = true;
    for (int p : points_to_remove) {
        if (!first_point) {
            std::cout << ", ";
        }
        std::cout << p;
        first_point = false;
    }
    std::cout << "]" << std::endl;
    int num_triangles_to_remove =
        std::count(triangle_to_remove.begin(), triangle_to_remove.end(), true);
    std::cout << "  Removing " << num_triangles_to_remove << " old refined triangles:" << std::endl;
    for (size_t i = 0; i < triangle_to_remove.size(); ++i) {
        if (triangle_to_remove[i]) {
            const auto& tri = surface.query_triangles[i];
            int tet_id = (i < surface.tet_ids.size()) ? surface.tet_ids[i] : -1;
            std::cout << "    Removing triangle [" << tri(0) << "," << tri(1) << "," << tri(2)
                      << "] with tet_id " << tet_id << std::endl;
        }
    }
    std::vector<int> old_to_new_point_map(surface.points.size(), -1);
    int new_point_idx = 0;
    for (size_t i = 0; i < surface.points.size(); ++i) {
        if (points_to_remove.find(static_cast<int>(i)) == points_to_remove.end()) {
            old_to_new_point_map[i] = new_point_idx++;
        }
    }
    std::vector<query_point_tet_r> new_points;
    new_points.reserve(new_point_idx);
    for (size_t i = 0; i < surface.points.size(); ++i) {
        if (points_to_remove.find(static_cast<int>(i)) == points_to_remove.end()) {
            new_points.push_back(surface.points[i]);
        }
    }
    std::vector<Eigen::Vector3i> final_triangles;
    final_triangles.reserve(surface.query_triangles.size() + new_triangles.size());
    std::vector<int> final_tet_ids;
    final_tet_ids.reserve(surface.tet_ids.size() + new_tet_ids.size());
    for (size_t i = 0; i < surface.query_triangles.size(); ++i) {
        if (!triangle_to_remove[i]) {
            const auto& tri = surface.query_triangles[i];
            Eigen::Vector3i new_tri;
            new_tri(0) = old_to_new_point_map[tri(0)];
            new_tri(1) = old_to_new_point_map[tri(1)];
            new_tri(2) = old_to_new_point_map[tri(2)];
            if (new_tri(0) >= 0 && new_tri(1) >= 0 && new_tri(2) >= 0) {
                final_triangles.push_back(new_tri);
                if (i < surface.tet_ids.size()) {
                    final_tet_ids.push_back(surface.tet_ids[i]);
                }
            }
        }
    }
    std::cout << "  Adding " << new_triangles.size() << " new triangles:" << std::endl;
    for (size_t i = 0; i < new_triangles.size(); ++i) {
        const auto& tri = new_triangles[i];
        Eigen::Vector3i new_tri;
        new_tri(0) = old_to_new_point_map[tri(0)];
        new_tri(1) = old_to_new_point_map[tri(1)];
        new_tri(2) = old_to_new_point_map[tri(2)];
        if (new_tri(0) >= 0 && new_tri(1) >= 0 && new_tri(2) >= 0) {
            int tet_id = (i < new_tet_ids.size()) ? new_tet_ids[i] : -1;
            std::cout << "    Adding triangle [" << new_tri(0) << "," << new_tri(1) << ","
                      << new_tri(2) << "] (before map: [" << tri(0) << "," << tri(1) << ","
                      << tri(2) << "]) with tet_id " << tet_id << std::endl;
            final_triangles.push_back(new_tri);
        }
    }
    for (int tet_id : new_tet_ids) {
        final_tet_ids.push_back(tet_id);
    }
    surface.points = std::move(new_points);
    surface.query_triangles = std::move(final_triangles);
    surface.tet_ids = std::move(final_tet_ids);
    std::cout << "  Final surface after simplification: " << surface.points.size() << " points, "
              << surface.query_triangles.size() << " triangles" << std::endl;

    // check manifold property after simplification
    std::map<std::pair<int, int>, int> final_edge_count;
    for (const auto& tri : surface.query_triangles) {
        for (int j = 0; j < 3; ++j) {
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            final_edge_count[{v0, v1}]++;
        }
    }
    int non_manifold_edges = 0;
    for (const auto& [edge, count] : final_edge_count) {
        if (count > 2) {
            non_manifold_edges++;
            std::cout << "    Warning: Non-manifold edge (" << edge.first << ", " << edge.second
                      << ") appears " << count << " times" << std::endl;
        }
    }
    if (non_manifold_edges > 0) {
        std::cout << "  Warning: Found " << non_manifold_edges
                  << " non-manifold edges after simplification" << std::endl;
    } else {
        std::cout << "  Surface is edge-manifold after simplification" << std::endl;
    }
    std::cout << "=== Refined triangles simplification completed ===" << std::endl;
}

} // namespace tet_surface_tracking_with_connectivity

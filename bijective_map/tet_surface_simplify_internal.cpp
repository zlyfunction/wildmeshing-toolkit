#include "tet_surface_simplify_internal.hpp"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/number_utils.h>
#include <gmp.h>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include "tet_track_operations.hpp"
#include "vtu_utils.hpp"

namespace PMP = CGAL::Polygon_mesh_processing;

namespace tet_surface_tracking_with_connectivity {

namespace {
// CGAL types for self-intersection check with exact rational arithmetic
using RationalKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using RationalPoint = RationalKernel::Point_3;
using CgalTriangle = std::array<std::size_t, 3>;

RationalKernel::FT rational_to_gmpq(const wmtk::Rational& r)
{
    mpq_t q;
    mpq_init(q);
    r.export_mpq(q);
    using ET = RationalKernel::FT::ET;
    ET et_expr(q);
    RationalKernel::FT result(et_expr);
    mpq_clear(q);
    return result;
}

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
        // If there is only one or zero interior points, no need to simplify
        if (interior_points.size() <= 1) {
            std::cout << "    Only " << interior_points.size()
                      << " interior points, skipping simplification for tet_id " << tet_id
                      << std::endl;
            continue;
        }

        if (boundary_edges.empty()) {
            throw std::runtime_error(
                "Error: simplify_refined_triangles_by_tet: no boundary edges found for tet_id " +
                std::to_string(tet_id));
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
        std::vector<Eigen::Vector3i> all_triangulated_triangles;
        std::set<size_t> triangles_to_remove_set;
        std::set<int>
            interior_points_to_remove; // Only interior points that are part of fan triangulation
        std::set<int> visited_interior;
        std::vector<int>
            new_centroid_point_indices; // Track newly added centroid points for rollback
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
            if (component_interior_points.size() == 1) {
                std::cout
                    << "      Component has only one interior point (already fan shape), skipping"
                    << std::endl;
                continue;
            }
            // For each interior point in the component, look at its triangles
            // For each triangle, find the opposite edge (the edge not containing the interior
            // point) If both endpoints of the opposite edge are boundary points, add to loop
            std::set<std::pair<int, int>> component_boundary_edges;
            std::set<size_t> component_triangles;
            // Debug: count triangles by number of interior points they contain
            int tri_with_1_interior = 0;
            int tri_with_2_interior = 0;
            int tri_with_3_interior = 0;
            std::vector<std::tuple<size_t, int, int, int>> tris_with_2plus_interior;
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
                }
            }
            // Analyze all component triangles
            for (size_t tri_idx : component_triangles) {
                const auto& tri = surface.query_triangles[tri_idx];
                int interior_count = 0;
                for (int j = 0; j < 3; ++j) {
                    if (component_interior_points.find(tri(j)) != component_interior_points.end()) {
                        interior_count++;
                    }
                }
                if (interior_count == 1) {
                    tri_with_1_interior++;
                } else if (interior_count == 2) {
                    tri_with_2_interior++;
                    tris_with_2plus_interior.push_back({tri_idx, tri(0), tri(1), tri(2)});
                } else if (interior_count == 3) {
                    tri_with_3_interior++;
                    tris_with_2plus_interior.push_back({tri_idx, tri(0), tri(1), tri(2)});
                }
            }
            std::cout << "      Triangle analysis: " << tri_with_1_interior << " with 1 interior, "
                      << tri_with_2_interior << " with 2 interior, " << tri_with_3_interior
                      << " with 3 interior" << std::endl;
            if (!tris_with_2plus_interior.empty()) {
                std::cout
                    << "      Triangles with 2+ interior points (may cause disconnected loops):"
                    << std::endl;
                for (const auto& t : tris_with_2plus_interior) {
                    std::cout << "        tri[" << std::get<0>(t) << "]: (" << std::get<1>(t)
                              << ", " << std::get<2>(t) << ", " << std::get<3>(t) << ")"
                              << std::endl;
                }
            }
            // Now collect boundary edges
            for (size_t tri_idx : component_triangles) {
                const auto& tri = surface.query_triangles[tri_idx];
                // Find all edges where both endpoints are boundary points
                for (int j = 0; j < 3; ++j) {
                    int v0 = tri(j);
                    int v1 = tri((j + 1) % 3);
                    if (boundary_points.find(v0) != boundary_points.end() &&
                        boundary_points.find(v1) != boundary_points.end()) {
                        std::pair<int, int> edge =
                            (v0 < v1) ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
                        component_boundary_edges.insert(edge);
                    }
                }
            }
            // Print component_boundary_edges
            std::cout << "      Component boundary edges (" << component_boundary_edges.size()
                      << " edges): [";
            bool first_edge = true;
            for (const auto& edge : component_boundary_edges) {
                if (!first_edge) {
                    std::cout << ", ";
                }
                std::cout << "(" << edge.first << "-" << edge.second << ")";
                first_edge = false;
            }
            std::cout << "]" << std::endl;
            // Check if edges form a valid loop (each vertex should appear exactly twice)
            std::map<int, int> vertex_degree;
            for (const auto& edge : component_boundary_edges) {
                vertex_degree[edge.first]++;
                vertex_degree[edge.second]++;
            }
            bool is_valid_loop = true;
            std::vector<int> invalid_vertices;
            for (const auto& vd : vertex_degree) {
                if (vd.second != 2) {
                    is_valid_loop = false;
                    invalid_vertices.push_back(vd.first);
                }
            }
            if (!is_valid_loop) {
                std::cout << "      WARNING: Boundary edges do NOT form a valid loop!" << std::endl;
                std::cout << "        Vertices with invalid degree:" << std::endl;
                for (int v : invalid_vertices) {
                    std::cout << "          Vertex " << v << " has degree " << vertex_degree[v]
                              << " (expected 2)" << std::endl;
                }
                std::cout << "        Skipping fan triangulation for this component" << std::endl;
                continue;
            }
            // Check if edges form a single connected loop (not multiple disconnected loops)
            if (!component_boundary_edges.empty()) {
                std::map<int, std::set<int>> adj;
                for (const auto& edge : component_boundary_edges) {
                    adj[edge.first].insert(edge.second);
                    adj[edge.second].insert(edge.first);
                }
                std::set<int> visited_vertices;
                std::queue<int> bfs_q;
                int start_v = adj.begin()->first;
                bfs_q.push(start_v);
                visited_vertices.insert(start_v);
                while (!bfs_q.empty()) {
                    int curr = bfs_q.front();
                    bfs_q.pop();
                    for (int neighbor : adj[curr]) {
                        if (visited_vertices.find(neighbor) == visited_vertices.end()) {
                            visited_vertices.insert(neighbor);
                            bfs_q.push(neighbor);
                        }
                    }
                }
                if (visited_vertices.size() != adj.size()) {
                    std::cout << "      WARNING: Boundary edges form multiple disconnected loops!"
                              << std::endl;
                    std::cout << "        Connected: " << visited_vertices.size()
                              << " vertices, Total: " << adj.size() << " vertices" << std::endl;
                    std::cout << "        Skipping fan triangulation for this component"
                              << std::endl;
                    continue;
                }
                std::cout << "      Boundary edges form a valid single loop with "
                          << vertex_degree.size() << " vertices" << std::endl;
            }
            // Compute centroid of all interior points using rational arithmetic
            // Sum up all barycentric coordinates (they must be in the same tet)
            Eigen::Matrix<wmtk::Rational, 4, 1> sum_bc;
            sum_bc.setZero();
            int64_t ref_tet_id = -1;
            Eigen::Vector4i ref_tv_ids;
            int num_interior_points = 0;
            for (int interior_p_idx : component_interior_points) {
                const auto& qp = surface.points[interior_p_idx];
                if (ref_tet_id == -1) {
                    ref_tet_id = qp.t_id;
                    ref_tv_ids = qp.tv_ids;
                }
                // Add barycentric coordinates
                for (int i = 0; i < 4; ++i) {
                    sum_bc(i) += qp.bc(i);
                }
                num_interior_points++;
            }
            if (num_interior_points == 0) {
                std::cout << "      No interior points to average, skipping" << std::endl;
                continue;
            }
            // Compute average: divide by number of points
            wmtk::Rational num_points_r(num_interior_points);
            Eigen::Matrix<wmtk::Rational, 4, 1> avg_bc;
            for (int i = 0; i < 4; ++i) {
                avg_bc(i) = sum_bc(i) / num_points_r;
            }
            // Normalize barycentric coordinates exactly using rational arithmetic
            wmtk::Rational bc_sum = avg_bc(0) + avg_bc(1) + avg_bc(2) + avg_bc(3);
            Eigen::Matrix<wmtk::Rational, 4, 1> new_bc_rational;
            if (bc_sum != wmtk::Rational(0)) {
                for (int i = 0; i < 4; ++i) {
                    new_bc_rational(i) = avg_bc(i) / bc_sum;
                }
            } else {
                // Fallback to center of tet
                wmtk::Rational quarter(1, 4);
                for (int i = 0; i < 4; ++i) {
                    new_bc_rational(i) = quarter;
                }
            }
            std::cout << "      Computed centroid bc (rational): ["
                      << new_bc_rational(0).to_double() << ", " << new_bc_rational(1).to_double()
                      << ", " << new_bc_rational(2).to_double() << ", "
                      << new_bc_rational(3).to_double() << "]" << std::endl;
            // Create new point with exact rational coordinates
            query_point_tet_r new_point;
            new_point.t_id = ref_tet_id;
            new_point.tv_ids = ref_tv_ids;
            new_point.bc = new_bc_rational;
            int new_point_idx = surface.points.size();
            surface.points.push_back(new_point);
            new_centroid_point_indices.push_back(new_point_idx);
            std::cout << "      Created new point at index " << new_point_idx << std::endl;
            // Create fan triangles: for each boundary edge, create a triangle with the new point
            std::cout << "      Creating fan triangles from " << component_boundary_edges.size()
                      << " boundary edges" << std::endl;
            for (const auto& edge : component_boundary_edges) {
                Eigen::Vector3i fan_tri;
                fan_tri(0) = edge.first;
                fan_tri(1) = edge.second;
                fan_tri(2) = new_point_idx;
                all_triangulated_triangles.push_back(fan_tri);
                std::cout << "        Fan triangle: [" << fan_tri(0) << "," << fan_tri(1) << ","
                          << fan_tri(2) << "]" << std::endl;
            }
            // Mark triangles incident to this component for removal
            for (size_t tri_idx : component_triangles) {
                triangles_to_remove_set.insert(tri_idx);
            }
            // Mark interior points of this component for removal (only if fan triangulation was
            // performed)
            for (int interior_p : component_interior_points) {
                interior_points_to_remove.insert(interior_p);
            }
        }
        if (all_triangulated_triangles.empty()) {
            std::cout << "    No fan triangles created, skipping tet_id " << tet_id << std::endl;
            continue;
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
                  << " new triangles from fan triangulation, "
                  << (all_final_triangles.size() - all_triangulated_triangles.size())
                  << " preserved triangles, total: " << all_final_triangles.size() << " triangles"
                  << std::endl;
        // Self-intersection check using CGAL with exact rational arithmetic
        // Use first 3 components of bc as positions
        {
            // Collect unique points and build position map using bc(0:3)
            std::set<int> check_points;
            for (const auto& tri : all_final_triangles) {
                check_points.insert(tri(0));
                check_points.insert(tri(1));
                check_points.insert(tri(2));
            }
            std::map<int, std::size_t> point_to_cgal_idx;
            std::vector<RationalPoint> cgal_points;
            cgal_points.reserve(check_points.size());
            for (int p_idx : check_points) {
                point_to_cgal_idx[p_idx] = cgal_points.size();
                const auto& qp = surface.points[p_idx];
                // Use first 3 barycentric coordinates as position with exact rational arithmetic
                RationalKernel::FT x = rational_to_gmpq(qp.bc(0));
                RationalKernel::FT y = rational_to_gmpq(qp.bc(1));
                RationalKernel::FT z = rational_to_gmpq(qp.bc(2));
                cgal_points.emplace_back(x, y, z);
            }
            // Build triangle soup
            std::vector<CgalTriangle> cgal_triangles;
            cgal_triangles.reserve(all_final_triangles.size());
            for (const auto& tri : all_final_triangles) {
                cgal_triangles.push_back(CgalTriangle{
                    point_to_cgal_idx[tri(0)],
                    point_to_cgal_idx[tri(1)],
                    point_to_cgal_idx[tri(2)]});
            }
            // Check for self-intersection
            bool has_self_intersection =
                PMP::does_triangle_soup_self_intersect(cgal_points, cgal_triangles);
            if (has_self_intersection) {
                std::cout
                    << "    WARNING: Self-intersection detected after simplification for tet_id "
                    << tet_id << ". Rolling back simplification." << std::endl;
                // Rollback: remove newly added centroid points from surface.points
                // Remove in reverse order to keep indices valid
                std::sort(
                    new_centroid_point_indices.begin(),
                    new_centroid_point_indices.end(),
                    std::greater<int>());
                for (int remove_idx : new_centroid_point_indices) {
                    if (remove_idx >= 0 && remove_idx < static_cast<int>(surface.points.size())) {
                        surface.points.erase(surface.points.begin() + remove_idx);
                        std::cout << "      Removed centroid point at index " << remove_idx
                                  << std::endl;
                    }
                }
                std::cout << "    Skipping tet_id " << tet_id << " due to self-intersection"
                          << std::endl;
                continue; // Skip this tet's simplification
            }
            std::cout << "    Self-intersection check passed for tet_id " << tet_id << std::endl;
        }
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
        // Only remove interior points that were part of fan triangulation
        for (int p : interior_points_to_remove) {
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

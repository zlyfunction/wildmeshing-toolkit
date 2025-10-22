#include "tet_surface_tracking.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <limits>
#include "tet_track_operations.hpp"
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

void check_manifold_property(
    const Eigen::MatrixXd& surface_V,
    const Eigen::MatrixXi& surface_F)
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

    std::cout << "Original mesh: " << surface_V.rows() << " vertices, "
              << surface_F.rows() << " faces" << std::endl;
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
            std::cout << "    Non-manifold edge: (" << edge.first.first << ", "
                      << edge.first.second << ") appears " << edge.second << " times"
                      << std::endl;
            std::cout << "    Edge vertices positions: ("
                      << V_triangle.row(edge.first.first) << ", "
                      << V_triangle.row(edge.first.second) << ")" << std::endl;

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
                          << face(1) << ", " << face(2) << "), area = " << area
                          << std::endl;
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
        std::cout << "  Non-manifold vertices: " << non_manifold_vertices.size()
                  << std::endl;

        // Create point mesh for non-manifold vertices
        Eigen::MatrixXd V_non_manifold(non_manifold_vertices.size(), 3);
        int idx = 0;
        for (int vertex_id : non_manifold_vertices) {
            V_non_manifold.row(idx) = V_triangle.row(vertex_id);
            idx++;
        }

        // Write non-manifold vertices to file
        vtu_utils::write_point_mesh_to_vtu(V_non_manifold, "non_manifold_vertices.vtu");
        std::cout << "  Non-manifold vertices written to non_manifold_vertices.vtu"
                  << std::endl;
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
        std::cerr << "Error: Surface file not found. Please provide a valid surface file." << std::endl;
        return;
    } else {
        std::cout << "query_surface found, reading from file..." << std::endl;
        query_surface = read_query_surface_tet_from_file(query_surface_filename);
    }

    auto [surface_V, surface_F] = query_surface_to_world_positions(query_surface, V_after);
    vtu_utils::write_triangle_mesh_to_vtu(surface_V, surface_F, "query_surface_tet_after.vtu");

    std::cout << "before tracking, surface size: " << query_surface.triangles.size()
              << std::endl;
    track_surface_tet(operation_logs_dir, query_surface, false, false);
    std::cout << "after tracking, surface size: " << query_surface.triangles.size()
              << std::endl;

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

} // namespace tet_surface_tracking

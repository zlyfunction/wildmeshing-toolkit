#include "tet_surface_simplify_internal.hpp"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/number_utils.h>
#include <gmp.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
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

// Point location classification based on barycentric coordinates
enum class PointLocation {
    Interior, // All bc non-zero (inside tet)
    OnFace, // One bc is zero (on a face)
    OnEdge, // Two bc are zero (on an edge)
    OnVertex // Three bc are zero (on a vertex)
};

struct PointClassification
{
    PointLocation location;
    // Global vertex ids of the tet where bc != 0 (constraint set)
    std::set<int> constraint_vids;
};

// Classify a point based on its barycentric coordinates
PointClassification classify_point(const query_point_tet_r& point)
{
    PointClassification result;
    result.constraint_vids.clear();
    for (int i = 0; i < 4; ++i) {
        if (point.bc(i) != wmtk::Rational(0)) {
            result.constraint_vids.insert(point.tv_ids(i));
        }
    }
    size_t num_nonzeros = result.constraint_vids.size();
    if (num_nonzeros == 4) {
        result.location = PointLocation::Interior;
    } else if (num_nonzeros == 3) {
        result.location = PointLocation::OnFace;
    } else if (num_nonzeros == 2) {
        result.location = PointLocation::OnEdge;
    } else {
        result.location = PointLocation::OnVertex;
    }
    return result;
}

// Structure to hold local patch data
struct LocalPatch
{
    std::vector<size_t> triangle_indices; // Indices into surface.query_triangles
    std::set<int> all_points; // All point indices in this patch
    std::set<int> boundary_points; // Points on boundary edges
    std::map<std::pair<int, int>, int> edge_count; // Edge -> count
    std::vector<std::pair<int, int>> boundary_edges;
    std::map<int, PointClassification> point_classifications;
};

// Build local patch from triangles starting at start_tri_idx
LocalPatch build_local_patch(
    const query_surface_tet_with_connectivity& surface,
    size_t start_tri_idx)
{
    LocalPatch patch;
    // Collect all triangles from start_tri_idx to end
    for (size_t i = start_tri_idx; i < surface.query_triangles.size(); ++i) {
        patch.triangle_indices.push_back(i);
    }
    // Collect all points and count edges
    for (size_t tri_idx : patch.triangle_indices) {
        const auto& tri = surface.query_triangles[tri_idx];
        for (int j = 0; j < 3; ++j) {
            patch.all_points.insert(tri(j));
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            patch.edge_count[{v0, v1}]++;
        }
    }
    // Find boundary edges and points
    for (const auto& [edge, count] : patch.edge_count) {
        if (count == 1) {
            patch.boundary_edges.push_back(edge);
            patch.boundary_points.insert(edge.first);
            patch.boundary_points.insert(edge.second);
        }
    }
    // Classify all points
    for (int p_idx : patch.all_points) {
        patch.point_classifications[p_idx] = classify_point(surface.points[p_idx]);
    }
    return patch;
}

// Check if edge collapse is valid based on point classifications
// Returns: 0 = cannot collapse, 1 = collapse to v0, 2 = collapse to v1
int can_collapse_edge(int v0, int v1, const LocalPatch& patch)
{
    const auto& class0 = patch.point_classifications.at(v0);
    const auto& class1 = patch.point_classifications.at(v1);
    bool v0_is_boundary = (patch.boundary_points.find(v0) != patch.boundary_points.end());
    bool v1_is_boundary = (patch.boundary_points.find(v1) != patch.boundary_points.end());
    const auto& c0 = class0.constraint_vids;
    const auto& c1 = class1.constraint_vids;
    auto is_subset = [](const std::set<int>& a, const std::set<int>& b) {
        return std::includes(b.begin(), b.end(), a.begin(), a.end());
    };
    // Rule 1: Both interior -> can collapse, keep first vertex
    if (class0.location == PointLocation::Interior && class1.location == PointLocation::Interior) {
        return 1; // Collapse to v0
    }
    // Rule 2: One on boundary face/edge/vertex, one interior -> collapse towards constrained point
    if (class0.location != PointLocation::Interior && class1.location == PointLocation::Interior) {
        return 1; // keep v0 (more constrained)
    }
    if (class0.location == PointLocation::Interior && class1.location != PointLocation::Interior) {
        return 2; // keep v1 (more constrained)
    }
    // Rule 3: Both constrained (face/edge/vertex)
    if ((class0.location == PointLocation::OnFace || class0.location == PointLocation::OnEdge ||
         class0.location == PointLocation::OnVertex) &&
        (class1.location == PointLocation::OnFace || class1.location == PointLocation::OnEdge ||
         class1.location == PointLocation::OnVertex)) {
        // Either is boundary point -> cannot collapse
        if (v0_is_boundary || v1_is_boundary) {
            return 0;
        }
        // If one constraint set is subset of the other, collapse to the more constrained (smaller)
        bool c0_subset_c1 = is_subset(c0, c1);
        bool c1_subset_c0 = is_subset(c1, c0);
        if (c0_subset_c1 && c1_subset_c0) {
            // identical constraint sets -> keep v0
            return 1;
        }
        if (c0_subset_c1) {
            return 1; // keep more constrained (v0)
        }
        if (c1_subset_c0) {
            return 2; // keep more constrained (v1)
        }
        // Otherwise they don't share a consistent constraint -> cannot collapse
        return 0;
    }
    return 0;
}

// Compute world position from barycentric coordinates (double version for VTU output)
Eigen::Vector3d compute_world_position(
    const query_point_tet_r& point,
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before)
{
    Eigen::Matrix<double, 4, 3> tet_vertices;
    Eigen::Vector4d bc_double;
    for (int i = 0; i < 4; ++i) {
        bc_double(i) = point.bc(i).to_double();
    }
    // Map global tet ID to local index in T_before
    int local_tet_idx = -1;
    if (point.t_id >= 0) {
        auto it = std::find(id_map_before.begin(), id_map_before.end(), point.t_id);
        if (it != id_map_before.end()) {
            local_tet_idx = std::distance(id_map_before.begin(), it);
        }
    }
    if (local_tet_idx >= 0 && local_tet_idx < T_before.rows()) {
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
        // Fallback: use tv_ids and v_id_map_before
        for (int i = 0; i < 4; ++i) {
            if (bc_double(i) != 0.0) {
                int64_t global_v_id = point.tv_ids(i);
                auto it = std::find(v_id_map_before.begin(), v_id_map_before.end(), global_v_id);
                if (it != v_id_map_before.end()) {
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
    }
    return barycentric_to_world_tet<double>(bc_double, tet_vertices);
}

// Check for self-intersection using CGAL with world coordinates
bool check_self_intersection(
    const std::vector<query_point_tet_r>& points,
    const std::vector<Eigen::Vector3i>& triangles,
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before)
{
    if (triangles.empty()) return false;
    std::set<int> used_points;
    for (const auto& tri : triangles) {
        used_points.insert(tri(0));
        used_points.insert(tri(1));
        used_points.insert(tri(2));
    }
    std::map<int, std::size_t> point_to_cgal_idx;
    std::vector<RationalPoint> cgal_points;
    cgal_points.reserve(used_points.size());
    for (int p_idx : used_points) {
        point_to_cgal_idx[p_idx] = cgal_points.size();
        const auto& qp = points[p_idx];
        // Compute world position using barycentric coordinates and tet vertices
        Eigen::Vector3d world_pos =
            compute_world_position(qp, V_before, T_before, id_map_before, v_id_map_before);
        RationalKernel::FT x(world_pos(0));
        RationalKernel::FT y(world_pos(1));
        RationalKernel::FT z(world_pos(2));
        cgal_points.emplace_back(x, y, z);
    }
    std::vector<CgalTriangle> cgal_triangles;
    cgal_triangles.reserve(triangles.size());
    for (const auto& tri : triangles) {
        cgal_triangles.push_back(CgalTriangle{
            point_to_cgal_idx[tri(0)],
            point_to_cgal_idx[tri(1)],
            point_to_cgal_idx[tri(2)]});
    }
    return PMP::does_triangle_soup_self_intersect(cgal_points, cgal_triangles);
}

// Perform edge collapse: replace v_remove with v_keep in all triangles
// Remove degenerate triangles (those with duplicate vertices)
void perform_edge_collapse(
    std::vector<Eigen::Vector3i>& triangles,
    std::vector<int>& tet_ids,
    int v_remove,
    int v_keep)
{
    std::vector<Eigen::Vector3i> new_triangles;
    std::vector<int> new_tet_ids;
    new_triangles.reserve(triangles.size());
    new_tet_ids.reserve(tet_ids.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        Eigen::Vector3i tri = triangles[i];
        // Replace v_remove with v_keep
        for (int j = 0; j < 3; ++j) {
            if (tri(j) == v_remove) {
                tri(j) = v_keep;
            }
        }
        // Check if degenerate (duplicate vertices)
        if (tri(0) != tri(1) && tri(1) != tri(2) && tri(0) != tri(2)) {
            new_triangles.push_back(tri);
            if (i < tet_ids.size()) {
                new_tet_ids.push_back(tet_ids[i]);
            }
        }
    }
    triangles = std::move(new_triangles);
    tet_ids = std::move(new_tet_ids);
}

// Get all edges from triangles
std::vector<std::pair<int, int>> get_all_edges(const std::vector<Eigen::Vector3i>& triangles)
{
    std::set<std::pair<int, int>> edge_set;
    for (const auto& tri : triangles) {
        for (int j = 0; j < 3; ++j) {
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            edge_set.insert({v0, v1});
        }
    }
    return std::vector<std::pair<int, int>>(edge_set.begin(), edge_set.end());
}

// Update local patch after collapse
void update_patch_after_collapse(
    LocalPatch& patch,
    const std::vector<Eigen::Vector3i>& triangles,
    const query_surface_tet_with_connectivity& surface)
{
    // Rebuild edge count
    patch.edge_count.clear();
    patch.all_points.clear();
    for (const auto& tri : triangles) {
        for (int j = 0; j < 3; ++j) {
            patch.all_points.insert(tri(j));
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            patch.edge_count[{v0, v1}]++;
        }
    }
    // Rebuild boundary info
    patch.boundary_edges.clear();
    patch.boundary_points.clear();
    for (const auto& [edge, count] : patch.edge_count) {
        if (count == 1) {
            patch.boundary_edges.push_back(edge);
            patch.boundary_points.insert(edge.first);
            patch.boundary_points.insert(edge.second);
        }
    }
    // Update point classifications for new points only
    for (int p_idx : patch.all_points) {
        if (patch.point_classifications.find(p_idx) == patch.point_classifications.end()) {
            patch.point_classifications[p_idx] = classify_point(surface.points[p_idx]);
        }
    }
}

// Write local patch triangles to VTU file
void write_local_patch_to_vtu(
    const std::vector<query_point_tet_r>& all_points,
    const std::vector<Eigen::Vector3i>& triangles,
    const std::vector<int>& tet_ids,
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const std::string& filename)
{
    if (triangles.empty()) {
        std::cout << "    Skipping VTU write: empty triangles" << std::endl;
        return;
    }
    // Collect unique points
    std::set<int> used_point_indices;
    for (const auto& tri : triangles) {
        used_point_indices.insert(tri(0));
        used_point_indices.insert(tri(1));
        used_point_indices.insert(tri(2));
    }
    // Build local index mapping
    std::map<int, int> global_to_local;
    std::vector<int> local_to_global;
    int local_idx = 0;
    for (int p_idx : used_point_indices) {
        global_to_local[p_idx] = local_idx++;
        local_to_global.push_back(p_idx);
    }
    // Build V matrix with world positions
    Eigen::MatrixXd V(local_to_global.size(), 3);
    Eigen::VectorXi point_global_ids(local_to_global.size());
    for (size_t i = 0; i < local_to_global.size(); ++i) {
        int global_idx = local_to_global[i];
        const auto& qp = all_points[global_idx];
        Eigen::Vector3d world_pos =
            compute_world_position(qp, V_before, T_before, id_map_before, v_id_map_before);
        V.row(i) = world_pos.transpose();
        point_global_ids(i) = global_idx;
    }
    // Build F matrix with local indices
    Eigen::MatrixXi F(triangles.size(), 3);
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        F(i, 0) = global_to_local[tri(0)];
        F(i, 1) = global_to_local[tri(1)];
        F(i, 2) = global_to_local[tri(2)];
    }
    // Build tet_id array for triangles
    Eigen::VectorXi tri_tet_ids(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        tri_tet_ids(i) = (i < tet_ids.size()) ? tet_ids[i] : -1;
    }
    // Write to VTU
    vtu_utils::write_triangle_mesh_to_vtu(V, F, filename, &tri_tet_ids, "tet_id");
    std::cout << "    Saved local patch to: " << filename << " (" << V.rows() << " points, "
              << F.rows() << " triangles)" << std::endl;
}

// Check if a triangle list is manifold
bool check_manifold(const std::vector<Eigen::Vector3i>& triangles)
{
    if (triangles.empty()) return true;
    Eigen::MatrixXi F(triangles.size(), 3);
    for (size_t i = 0; i < triangles.size(); ++i) {
        F.row(i) = triangles[i];
    }
    // for debug, print the triangles
    {
        std::cout << "Triangle:[ " << std::endl;
        for (size_t i = 0; i < triangles.size(); ++i) {
            const auto& tri = triangles[i];
            std::cout << "[" << tri(0) << ", " << tri(1) << ", " << tri(2) << "] ";
            if (i != triangles.size() - 1) {
                std::cout << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    bool is_edge_manifold = igl::is_edge_manifold(F);
    if (!is_edge_manifold) {
        std::cout << "Surface is not edge manifold" << std::endl;
        return false;
    }
    bool is_vertex_manifold = igl::is_vertex_manifold(F);
    if (!is_vertex_manifold) {
        std::cout << "Surface is not vertex manifold" << std::endl;
        return false;
    }
    return true;
}

} // namespace

// Simplify refined triangles using edge collapse
void simplify_refined_triangles_by_tet(
    query_surface_tet_with_connectivity& surface,
    size_t start_tri_idx,
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    int operation_id)
{
    std::cout << "\n=== Starting edge collapse simplification ===" << std::endl;
    std::cout << "  Processing triangles from index " << start_tri_idx << " to "
              << surface.query_triangles.size() << std::endl;
    if (start_tri_idx >= surface.query_triangles.size()) {
        std::cout << "  No refined triangles to simplify" << std::endl;
        return;
    }
    size_t num_refined_triangles = surface.query_triangles.size() - start_tri_idx;
    std::cout << "  Found " << num_refined_triangles << " refined triangles to process"
              << std::endl;
    // Step 1: Build local patch with boundary information
    std::cout << "\n  Step 1: Building local patch..." << std::endl;
    LocalPatch patch = build_local_patch(surface, start_tri_idx);
    std::cout << "    Total points: " << patch.all_points.size() << std::endl;
    std::cout << "    Boundary points: " << patch.boundary_points.size() << std::endl;
    std::cout << "    Boundary edges: " << patch.boundary_edges.size() << std::endl;
    // Step 2: Print point classifications
    std::cout << "\n  Step 2: Point classifications:" << std::endl;
    int num_interior = 0, num_on_face = 0, num_on_edge = 0, num_on_vertex = 0;
    for (const auto& [p_idx, classification] : patch.point_classifications) {
        switch (classification.location) {
        case PointLocation::Interior: num_interior++; break;
        case PointLocation::OnFace: num_on_face++; break;
        case PointLocation::OnEdge: num_on_edge++; break;
        case PointLocation::OnVertex: num_on_vertex++; break;
        }
    }
    std::cout << "    Interior: " << num_interior << ", OnFace: " << num_on_face
              << ", OnEdge: " << num_on_edge << ", OnVertex: " << num_on_vertex << std::endl;
    // Copy triangles and tet_ids for this patch to work with
    std::vector<Eigen::Vector3i> working_triangles;
    std::vector<int> working_tet_ids;
    for (size_t tri_idx : patch.triangle_indices) {
        working_triangles.push_back(surface.query_triangles[tri_idx]);
        if (tri_idx < surface.tet_ids.size()) {
            working_tet_ids.push_back(surface.tet_ids[tri_idx]);
        }
    }
    // Write local patch BEFORE simplification
    std::string before_filename = "simplify_before_op" + std::to_string(operation_id) + ".vtu";
    write_local_patch_to_vtu(
        surface.points,
        working_triangles,
        working_tet_ids,
        V_before,
        T_before,
        id_map_before,
        v_id_map_before,
        before_filename);
    // Step 3: Edge collapse loop
    std::cout << "\n  Step 3: Starting edge collapse..." << std::endl;
    int total_collapses = 0;
    int max_iterations = static_cast<int>(patch.all_points.size() * 2);
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto edges = get_all_edges(working_triangles);
        bool collapsed_any = false;
        for (const auto& [v0, v1] : edges) {
            int collapse_direction = can_collapse_edge(v0, v1, patch);
            if (collapse_direction == 0) continue;
            int v_keep = (collapse_direction == 1) ? v0 : v1;
            int v_remove = (collapse_direction == 1) ? v1 : v0;
            // Save state for rollback
            auto backup_triangles = working_triangles;
            auto backup_tet_ids = working_tet_ids;
            // Perform collapse
            perform_edge_collapse(working_triangles, working_tet_ids, v_remove, v_keep);
            // Check self-intersection
            if (check_self_intersection(
                    surface.points,
                    working_triangles,
                    V_before,
                    T_before,
                    id_map_before,
                    v_id_map_before)) {
                std::cout << "    Rollback: edge (" << v0 << "," << v1
                          << ") collapse caused self-intersection" << std::endl;
                working_triangles = std::move(backup_triangles);
                working_tet_ids = std::move(backup_tet_ids);
                continue;
            }
            std::cout << "    Collapsed edge (" << v0 << "," << v1 << ") -> keep " << v_keep
                      << std::endl;
            total_collapses++;
            collapsed_any = true;
            // Update patch for next iteration
            update_patch_after_collapse(patch, working_triangles, surface);
            break; // Restart edge iteration
        }
        if (!collapsed_any) {
            std::cout << "    No more edges can be collapsed" << std::endl;
            break;
        }
    }
    std::cout << "    Total collapses performed: " << total_collapses << std::endl;
    // Write local patch AFTER simplification
    std::string after_filename = "simplify_after_op" + std::to_string(operation_id) + ".vtu";
    write_local_patch_to_vtu(
        surface.points,
        working_triangles,
        working_tet_ids,
        V_before,
        T_before,
        id_map_before,
        v_id_map_before,
        after_filename);
    if (total_collapses == 0) {
        std::cout << "  No simplification performed, keeping original triangles" << std::endl;
        return;
    }
    // Step 4: Update surface with simplified triangles (remove unreferenced points first)
    std::cout << "\n  Step 4: Updating surface..." << std::endl;
    // Points used before the patch (must be preserved)
    std::set<int> points_used_before_patch;
    for (size_t i = 0; i < start_tri_idx; ++i) {
        const auto& tri = surface.query_triangles[i];
        points_used_before_patch.insert(tri(0));
        points_used_before_patch.insert(tri(1));
        points_used_before_patch.insert(tri(2));
    }
    // Points used in simplified patch
    std::set<int> points_used_in_patch;
    for (const auto& tri : working_triangles) {
        points_used_in_patch.insert(tri(0));
        points_used_in_patch.insert(tri(1));
        points_used_in_patch.insert(tri(2));
    }
    // Build global used set: before-patch + simplified patch
    std::set<int> used_points_global = points_used_before_patch;
    used_points_global.insert(points_used_in_patch.begin(), points_used_in_patch.end());
    // Build remap for all points; remove unreferenced points globally
    std::vector<int> old_to_new(surface.points.size(), -1);
    int new_idx = 0;
    for (size_t i = 0; i < surface.points.size(); ++i) {
        if (used_points_global.find(static_cast<int>(i)) != used_points_global.end()) {
            old_to_new[i] = new_idx++;
        }
    }
    // Remap triangles before patch
    std::vector<Eigen::Vector3i> remapped_before_tris;
    remapped_before_tris.reserve(start_tri_idx);
    for (size_t i = 0; i < start_tri_idx; ++i) {
        const auto& tri = surface.query_triangles[i];
        Eigen::Vector3i new_tri(old_to_new[tri(0)], old_to_new[tri(1)], old_to_new[tri(2)]);
        if (new_tri(0) >= 0 && new_tri(1) >= 0 && new_tri(2) >= 0) {
            remapped_before_tris.push_back(new_tri);
        }
    }
    // Remap simplified patch triangles
    std::vector<Eigen::Vector3i> remapped_patch_tris;
    remapped_patch_tris.reserve(working_triangles.size());
    for (const auto& tri : working_triangles) {
        Eigen::Vector3i new_tri(old_to_new[tri(0)], old_to_new[tri(1)], old_to_new[tri(2)]);
        if (new_tri(0) >= 0 && new_tri(1) >= 0 && new_tri(2) >= 0) {
            remapped_patch_tris.push_back(new_tri);
        }
    }
    // Combined triangles for final manifold check (after removing unused points)
    std::vector<Eigen::Vector3i> combined_triangles = remapped_before_tris;
    combined_triangles.insert(
        combined_triangles.end(),
        remapped_patch_tris.begin(),
        remapped_patch_tris.end());
    if (!check_manifold(combined_triangles)) {
        throw std::runtime_error("Surface is not manifold after simplification.");
    }
    // Build new points vector
    std::vector<query_point_tet_r> new_points;
    new_points.reserve(new_idx);
    for (size_t i = 0; i < surface.points.size(); ++i) {
        if (old_to_new[i] >= 0) {
            new_points.push_back(surface.points[i]);
        }
    }
    // Build new triangles/tet_ids
    std::vector<Eigen::Vector3i> new_triangles;
    std::vector<int> new_tet_ids;
    // Add remapped before-patch triangles
    for (size_t i = 0; i < remapped_before_tris.size(); ++i) {
        new_triangles.push_back(remapped_before_tris[i]);
        if (i < surface.tet_ids.size()) {
            new_tet_ids.push_back(surface.tet_ids[i]);
        }
    }
    // Add remapped simplified patch triangles
    for (size_t i = 0; i < remapped_patch_tris.size(); ++i) {
        new_triangles.push_back(remapped_patch_tris[i]);
        if (i < working_tet_ids.size()) {
            new_tet_ids.push_back(working_tet_ids[i]);
        }
    }
    // Update surface
    surface.points = std::move(new_points);
    surface.query_triangles = std::move(new_triangles);
    surface.tet_ids = std::move(new_tet_ids);
    std::cout << "  Final surface: " << surface.points.size() << " points, "
              << surface.query_triangles.size() << " triangles" << std::endl;
    // Final manifold diagnostics (edge-only, optional)
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
    std::cout << "=== Edge collapse simplification completed ===" << std::endl;
}

} // namespace tet_surface_tracking_with_connectivity

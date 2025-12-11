#include "tet_surface_tracking_internal.hpp"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/number_utils.h>
#include <gmp.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include "batch_operation_log_reader.hpp"
#include "cgal_autorefine_utils_rational.hpp"
#include "tet_point_tracking.hpp"
#include "tet_surface_simplify_internal.hpp"
#include "tet_surface_tracking_with_connectivity.hpp"
#include "tet_track_operations.hpp"
#include "vtu_utils.hpp"

namespace PMP = CGAL::Polygon_mesh_processing;

namespace tet_surface_tracking_with_connectivity {

// Forward declaration for triangle sanity checks implemented in tet_surface_simplify_internal.cpp
void sanity_check_triangles(
    const query_surface_tet_with_connectivity& surface,
    const std::vector<int64_t>& id_map_before);

std::pair<MatrixXr, Eigen::MatrixXi> surface_to_world_positions_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V)
{
    std::cout << "Converting query surface with connectivity to world positions..." << std::endl;
    MatrixXr V_out(query_surface.points.size(), 3);
    Eigen::MatrixXi F_out(query_surface.query_triangles.size(), 3);
    for (size_t i = 0; i < query_surface.points.size(); i++) {
        const auto& pt = query_surface.points[i];
        Eigen::Vector4i tet_verts = pt.tv_ids;
        Vector3r world_pos = Vector3r::Zero();
        for (int j = 0; j < 4; j++) {
            world_pos += pt.bc(j) * V.row(tet_verts(j)).transpose();
        }
        V_out.row(i) = world_pos.transpose();
    }
    for (size_t i = 0; i < query_surface.query_triangles.size(); i++) {
        F_out.row(i) = query_surface.query_triangles[i];
    }
    std::cout << "Converted " << V_out.rows() << " vertices and " << F_out.rows() << " triangles"
              << std::endl;
    return {V_out, F_out};
}

bool check_surface_manifold_property(const std::vector<Eigen::Vector3i>& surface_F)
{
    Eigen::MatrixXi F(surface_F.size(), 3);
    for (size_t i = 0; i < surface_F.size(); i++) {
        F.row(i) = surface_F[i];
    }
    // Edge count diagnostics
    std::map<std::pair<int, int>, int> edge_count;
    for (const auto& tri : surface_F) {
        for (int j = 0; j < 3; ++j) {
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            edge_count[{v0, v1}]++;
        }
    }
    std::vector<std::pair<std::pair<int, int>, int>> bad_edges;
    for (const auto& [e, c] : edge_count) {
        if (c > 2) bad_edges.push_back({e, c});
    }

    bool is_edge_manifold_result = igl::is_edge_manifold(F);
    bool is_vertex_manifold_result = igl::is_vertex_manifold(F);
    if (!bad_edges.empty()) {
        std::cout << "Non-manifold edges (count > 2):" << std::endl;
        for (const auto& be : bad_edges) {
            std::cout << "  edge (" << be.first.first << "," << be.first.second
                      << ") count=" << be.second << std::endl;
        }
    }
    if (!is_edge_manifold_result) {
        std::cout << "Surface is not edge manifold (igl::is_edge_manifold=false)" << std::endl;
    }
    if (!is_vertex_manifold_result) {
        std::cout << "Surface is not vertex manifold (igl::is_vertex_manifold=false)" << std::endl;
    }
    return is_edge_manifold_result && is_vertex_manifold_result;
}

namespace {
using RationalKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using RationalPoint = RationalKernel::Point_3;
using Triangle = std::array<std::size_t, 3>;

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

std::vector<RationalPoint> rational_vertices_to_points(const MatrixXr& V)
{
    std::vector<RationalPoint> points;
    points.reserve(static_cast<std::size_t>(V.rows()));
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        RationalKernel::FT x = rational_to_gmpq(V(i, 0));
        RationalKernel::FT y = rational_to_gmpq(V(i, 1));
        RationalKernel::FT z = rational_to_gmpq(V(i, 2));
        points.emplace_back(x, y, z);
    }
    return points;
}
} // namespace

bool check_surface_self_intersection(
    const MatrixXr& surface_V,
    const std::vector<Eigen::Vector3i>& surface_F)
{
    std::vector<RationalPoint> points = rational_vertices_to_points(surface_V);
    std::vector<Triangle> triangles;
    triangles.reserve(surface_F.size());
    for (const auto& tri : surface_F) {
        triangles.push_back(Triangle{
            static_cast<std::size_t>(tri(0)),
            static_cast<std::size_t>(tri(1)),
            static_cast<std::size_t>(tri(2))});
    }
    return PMP::does_triangle_soup_self_intersect(points, triangles);
}

bool check_surface_self_intersection_intrinsic(
    const query_surface_tet_with_connectivity& query_surface,
    const Eigen::MatrixXi& T)
{
    // group all tet_ids that are in the query_surface by tet_id
    std::map<int, std::vector<int>> tet_ids_by_tet_id;
    for (int i = 0; i < query_surface.query_triangles.size(); i++) {
        int tet_id = query_surface.tet_ids[i];
        tet_ids_by_tet_id[tet_id].push_back(i);
    }
    bool has_self_intersection = false;
    // for each tet_id, check if it has self-intersection
    for (const auto& [tet_id, tri_ids] : tet_ids_by_tet_id) {
        // get local points and triangles
        std::vector<RationalPoint> local_points;
        std::vector<Triangle> local_triangles;
        std::map<int, int> global_to_local_point_map;

        auto tet_vids = T.row(tet_id);

        for (int tri_id : tri_ids) {
            const auto& tri = query_surface.query_triangles[tri_id];
            Eigen::Vector3i local_tri;
            for (int vi = 0; vi < 3; vi++) {
                int global_point_idx = tri[vi];

                if (global_to_local_point_map.find(global_point_idx) ==
                    global_to_local_point_map.end()) {
                    const auto& pt = query_surface.points[global_point_idx];
                    Eigen::Matrix<wmtk::Rational, 4, 1> real_bc;
                    real_bc.setZero();
                    for (int bc_idx = 0; bc_idx < 4; bc_idx++) {
                        if (pt.bc(bc_idx) != 0) {
                            int real_vid = pt.tv_ids(bc_idx);
                            auto it = std::find(tet_vids.begin(), tet_vids.end(), real_vid);
                            if (it != tet_vids.end()) {
                                real_bc(it - tet_vids.begin()) = pt.bc(bc_idx);
                            } else {
                                std::cout << "Error: non-zero bc vertex not found in tet_vids"
                                          << std::endl;
                                std::cout << "global_point_idx: " << global_point_idx << std::endl;
                                std::cout << "tet_vids: " << tet_vids.transpose() << std::endl;
                                std::cout << "pt.t_id: " << pt.t_id << std::endl;
                                std::cout << "pt.tv_ids: ";
                                for (int bc_print = 0; bc_print < 4; ++bc_print) {
                                    std::cout << pt.tv_ids(bc_print);
                                    if (bc_print < 3) std::cout << ", ";
                                }
                                std::cout << std::endl;
                                std::cout << "pt.bc: ";
                                for (int bc_print = 0; bc_print < 4; ++bc_print) {
                                    std::cout << pt.bc(bc_print).to_double();
                                    if (bc_print < 3) std::cout << ", ";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }

                    RationalKernel::FT x = rational_to_gmpq(real_bc(0));
                    RationalKernel::FT y = rational_to_gmpq(real_bc(1));
                    RationalKernel::FT z = rational_to_gmpq(real_bc(2));
                    local_points.emplace_back(x, y, z);
                    global_to_local_point_map[global_point_idx] = local_points.size() - 1;
                }

                local_tri[vi] = global_to_local_point_map[global_point_idx];
            }
            local_triangles.emplace_back(Triangle{
                static_cast<std::size_t>(local_tri[0]),
                static_cast<std::size_t>(local_tri[1]),
                static_cast<std::size_t>(local_tri[2])});
        }
        bool has_self_intersection_in_this_tet =
            PMP::does_triangle_soup_self_intersect(local_points, local_triangles);
        if (has_self_intersection_in_this_tet) {
            has_self_intersection = true;
            std::cout << "Self-intersection detected in tet_id " << tet_id << std::endl;
        }
    }

    return has_self_intersection;
}

void handle_consolidate_operation(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet_with_connectivity& surface,
    bool forward)
{
    std::cout << "Handling Consolidate operation for surface with connectivity" << std::endl;
    tet_point_tracking::handle_consolidate_tet<wmtk::Rational>(
        tet_ids_maps,
        vertex_ids_maps,
        surface.points,
        forward);
    if (!forward) {
        for (auto& tet_id : surface.tet_ids) {
            if (tet_id >= 0) {
                tet_id = tet_ids_maps[tet_id];
            }
        }
    } else {
        for (auto& tet_id : surface.tet_ids) {
            if (tet_id >= 0) {
                auto it = std::find(tet_ids_maps.begin(), tet_ids_maps.end(), tet_id);
                if (it != tet_ids_maps.end()) {
                    tet_id = std::distance(tet_ids_maps.begin(), it);
                }
            }
        }
    }
    std::cout << "Consolidate operation completed for " << surface.points.size() << " points and "
              << surface.tet_ids.size() << " triangle tet_ids" << std::endl;
}


void surface_triangle_arrangement(
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const std::vector<int64_t>& id_map_after,
    query_surface_tet_with_connectivity& surface,
    int operation_id,
    bool do_rounding,
    bool verbose,
    bool save_debug_meshes,
    bool do_simplify)
{
    // verbose = true;
    std::vector<int> face_ids;
    for (int i = 0; i < surface.query_triangles.size(); i++) {
        if (std::find(id_map_after.begin(), id_map_after.end(), surface.tet_ids[i]) !=
            id_map_after.end()) {
            face_ids.push_back(i);
        }
    }
    if (face_ids.size() == 0) {
        return;
    }
    if (verbose) {
        for (int idx = 0; idx < face_ids.size(); ++idx) {
            int face_id = face_ids[idx];
            const Eigen::Vector3i& tri = surface.query_triangles[face_id];
            std::cout << "Triangle " << idx << ":\n  [\n";
            std::cout << "    tet_id: " << surface.tet_ids[face_id] << std::endl;
            for (int vi = 0; vi < 3; ++vi) {
                int global_point_idx = tri[vi];
                const auto& pt = surface.points[global_point_idx];
                std::cout << "    { pt_idx: " << global_point_idx << ", tet_id: " << pt.t_id
                          << ", bc: [";
                for (int bc_i = 0; bc_i < 4; ++bc_i) {
                    std::cout << std::setprecision(16) << pt.bc[bc_i].to_double();
                    if (bc_i < 3) std::cout << ", ";
                }
                std::cout << "] }";
                if (vi < 2) std::cout << ",";
                std::cout << "\n";
            }
            std::cout << "  ]" << std::endl;
        }
    }
    std::set<int> unique_point_indices_set;
    for (int face_id : face_ids) {
        const Eigen::Vector3i& tri = surface.query_triangles[face_id];
        unique_point_indices_set.insert(tri[0]);
        unique_point_indices_set.insert(tri[1]);
        unique_point_indices_set.insert(tri[2]);
    }
    std::vector<int> unique_point_indices(
        unique_point_indices_set.begin(),
        unique_point_indices_set.end());
    std::map<int, int> global_to_local_point_map;
    for (int local_idx = 0; local_idx < unique_point_indices.size(); local_idx++) {
        global_to_local_point_map[unique_point_indices[local_idx]] = local_idx;
    }
    Eigen::MatrixXi local_triangles_F(face_ids.size(), 3);
    for (int i = 0; i < face_ids.size(); i++) {
        int face_id = face_ids[i];
        const Eigen::Vector3i& global_tri = surface.query_triangles[face_id];
        local_triangles_F(i, 0) = global_to_local_point_map[global_tri[0]];
        local_triangles_F(i, 1) = global_to_local_point_map[global_tri[1]];
        local_triangles_F(i, 2) = global_to_local_point_map[global_tri[2]];
    }
    if (verbose) {
        std::cout << "Built local triangles mesh: " << unique_point_indices.size() << " vertices, "
                  << local_triangles_F.rows() << " faces" << std::endl;
    }
    std::vector<cgal_autorefine_demo::SampledPointInputRational> sampled_points;
    sampled_points.reserve(unique_point_indices.size());
    for (int local_idx = 0; local_idx < unique_point_indices.size(); local_idx++) {
        int global_idx = unique_point_indices[local_idx];
        const auto& pt = surface.points[global_idx];
        cgal_autorefine_demo::SampledPointInputRational sampled_pt;
        sampled_pt.tet_index = -1;
        auto it = std::find(id_map_before.begin(), id_map_before.end(), pt.t_id);
        if (it != id_map_before.end()) {
            sampled_pt.tet_index = std::distance(id_map_before.begin(), it);
            sampled_pt.barycentric = pt.bc;
        } else {
            if (verbose) {
                std::cout << "Warning: tet_id " << pt.t_id << " not found in id_map_before"
                          << std::endl;
                std::cout
                    << "Starting to find the alternative representation for this point in this "
                       "local patch"
                    << std::endl;
            }
            {
                std::vector<int> non_zero_vids;
                std::vector<wmtk::Rational> non_zero_bcs;
                for (int bc_idx = 0; bc_idx < 4; bc_idx++) {
                    if (pt.bc(bc_idx) != 0) {
                        non_zero_vids.push_back(pt.tv_ids(bc_idx));
                        non_zero_bcs.push_back(pt.bc(bc_idx));
                    }
                }
                if (non_zero_vids.size() == 4) {
                    std::cout
                        << "This point does not have a valid representation in this local patch"
                        << std::endl;
                    throw std::runtime_error(
                        "This point does not have a valid representation in this local patch");
                }
                for (int tet_id = 0; tet_id < T_before.rows(); tet_id++) {
                    bool contains_all = true;
                    Eigen::Vector4i tet_mapped_vids;
                    tet_mapped_vids << v_id_map_before[T_before(tet_id, 0)],
                        v_id_map_before[T_before(tet_id, 1)], v_id_map_before[T_before(tet_id, 2)],
                        v_id_map_before[T_before(tet_id, 3)];
                    Vector4r bc_tmp = Vector4r::Zero();
                    for (int non_zero_vid_idx = 0; non_zero_vid_idx < non_zero_vids.size();
                         non_zero_vid_idx++) {
                        int non_zero_vid = non_zero_vids[non_zero_vid_idx];
                        wmtk::Rational non_zero_bc = non_zero_bcs[non_zero_vid_idx];
                        auto it =
                            std::find(tet_mapped_vids.begin(), tet_mapped_vids.end(), non_zero_vid);
                        if (it == tet_mapped_vids.end()) {
                            contains_all = false;
                            break;
                        }
                        bc_tmp(it - tet_mapped_vids.begin()) = non_zero_bc;
                    }
                    if (contains_all) {
                        sampled_pt.tet_index = tet_id;
                        sampled_pt.barycentric = bc_tmp;
                        break;
                    }
                }
            }
            if (sampled_pt.tet_index == -1) {
                std::cout << "Warning: failed to find the alternative representation for this "
                             "point in this local patch"
                          << std::endl;
                throw std::runtime_error("Failed to find the alternative representation for this "
                                         "point in this local patch");
            }
        }
        sampled_points.push_back(sampled_pt);
    }
    if (verbose) {
        std::cout << "Sampled Points: " << std::endl;
        for (size_t i = 0; i < sampled_points.size(); ++i) {
            const auto& sp = sampled_points[i];
            std::cout << "  [" << i << "] tet_index: " << sp.tet_index << ", barycentric: [";
            for (int j = 0; j < 4; ++j) {
                std::cout << sp.barycentric[j].to_double();
                if (j < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "local_triangles_F (faces):" << std::endl;
        for (int i = 0; i < local_triangles_F.rows(); ++i) {
            std::cout << "  [" << i << "]: ";
            for (int j = 0; j < 3; ++j) {
                std::cout << local_triangles_F(i, j);
                if (j < 2) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "Calling autorefine_sampled_triangles_rational on V_before and T_before..."
              << std::endl;
    auto autorefine_start = std::chrono::high_resolution_clock::now();
    cgal_autorefine_demo::AutorefineResultRational autorefine_result =
        cgal_autorefine_demo::autorefine_sampled_triangles_rational(
            V_before,
            T_before,
            sampled_points,
            local_triangles_F);
    auto autorefine_end = std::chrono::high_resolution_clock::now();
    auto autorefine_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(autorefine_end - autorefine_start);
    std::cout << "Autorefine completed: " << autorefine_result.refined_points.size()
              << " refined points, " << autorefine_result.refined_triangles.size()
              << " refined triangles" << std::endl;
    std::cout << "Autorefine took " << autorefine_duration.count() << " ms" << std::endl;
    std::cout << "Sampled fragment indices size: "
              << autorefine_result.sampled_fragment_indices.size() << std::endl;
    if (save_debug_meshes == true) {
        auto to_vertex_matrix = [](const std::vector<cgal_autorefine_demo::RationalPoint>& pts) {
            Eigen::MatrixXd V(pts.size(), 3);
            for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(pts.size()); ++i) {
                V(i, 0) = CGAL::to_double(pts[i].x());
                V(i, 1) = CGAL::to_double(pts[i].y());
                V(i, 2) = CGAL::to_double(pts[i].z());
            }
            return V;
        };
        auto to_face_matrix = [](const std::vector<cgal_autorefine_demo::Triangle>& tris) {
            Eigen::MatrixXi F(tris.size(), 3);
            for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(tris.size()); ++i) {
                F(i, 0) = static_cast<int>(tris[i][0]);
                F(i, 1) = static_cast<int>(tris[i][1]);
                F(i, 2) = static_cast<int>(tris[i][2]);
            }
            return F;
        };
        Eigen::MatrixXd V_original = to_vertex_matrix(autorefine_result.original_points);
        Eigen::MatrixXi F_original = to_face_matrix(autorefine_result.original_triangles);
        Eigen::MatrixXd V_refined = to_vertex_matrix(autorefine_result.refined_points);
        Eigen::MatrixXi F_refined = to_face_matrix(autorefine_result.refined_triangles);
        Eigen::VectorXi original_triangle_parent_vec =
            Eigen::VectorXi::Constant(autorefine_result.original_triangles.size(), -1);
        for (Eigen::Index i = 0; i < original_triangle_parent_vec.size(); ++i) {
            const auto& parents =
                autorefine_result.original_triangle_parent_tets[static_cast<std::size_t>(i)];
            if (!parents.empty()) {
                original_triangle_parent_vec(i) = parents.front();
            }
        }
        const Eigen::VectorXi& triangle_origin_ids = autorefine_result.origin_triangle_ids;
        const Eigen::VectorXi& triangle_origin_tet = autorefine_result.origin_tet_ids;
        std::string before_path =
            "surface_arrangement_op" + std::to_string(operation_id) + "_before.vtu";
        std::string after_path =
            "surface_arrangement_op" + std::to_string(operation_id) + "_after.vtu";
        std::string before_tet_path =
            "surface_arrangement_op" + std::to_string(operation_id) + "_before_tet.vtu";
        std::string after_tet_path =
            "surface_arrangement_op" + std::to_string(operation_id) + "_after_tet.vtu";
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
        if (local_triangles_F.rows() > 0 && local_triangles_F.rows() <= F_original.rows()) {
            Eigen::Index sampled_count = local_triangles_F.rows();
            Eigen::Index start_idx = F_original.rows() - sampled_count;
            Eigen::MatrixXi F_sampled(sampled_count, 3);
            for (Eigen::Index i = 0; i < sampled_count; ++i) {
                F_sampled.row(i) = F_original.row(start_idx + i);
            }
            std::string sampled_before_path =
                "surface_arrangement_op" + std::to_string(operation_id) + "_sampled_before.vtu";
            vtu_utils::write_triangle_mesh_to_vtu(V_original, F_sampled, sampled_before_path);
            std::cout << "  sampled (before refine) -> " << sampled_before_path << std::endl;
        }
        if (!autorefine_result.sampled_fragment_triangles.empty()) {
            Eigen::MatrixXi F_subset(autorefine_result.sampled_fragment_triangles.size(), 3);
            for (Eigen::Index i = 0; i < F_subset.rows(); ++i) {
                const auto& tri =
                    autorefine_result.sampled_fragment_triangles[static_cast<size_t>(i)];
                F_subset(i, 0) = static_cast<int>(tri[0]);
                F_subset(i, 1) = static_cast<int>(tri[1]);
                F_subset(i, 2) = static_cast<int>(tri[2]);
            }
            const Eigen::VectorXi& subset_tet_ids = autorefine_result.sampled_fragment_tet_ids;
            std::string sampled_path =
                "surface_arrangement_op" + std::to_string(operation_id) + "_refined_sampled.vtu";
            vtu_utils::write_triangle_mesh_to_vtu(
                V_refined,
                F_subset,
                sampled_path,
                subset_tet_ids.size() == F_subset.rows() ? &subset_tet_ids : nullptr,
                "sampled_tet_id");
            std::cout << "  refined sampled only -> " << sampled_path << std::endl;
        }
        std::cout << "Saved arrangement VTUs:\n  initial soup -> " << before_path
                  << "\n  initial soup (origin_tet_id) -> " << before_tet_path
                  << "\n  refined soup -> " << after_path << "\n  refined soup (origin_tet_id) -> "
                  << after_tet_path << std::endl;
    }
    if (verbose) {
        if (!autorefine_result.sampled_fragment_triangles.empty()) {
            std::cout << "\n=== Refined Sampled Triangles ===" << std::endl;
            std::cout << "Number of refined sampled triangle fragments: "
                      << autorefine_result.sampled_fragment_triangles.size() << std::endl;
            std::set<std::size_t> sampled_vertex_ids;
            for (std::size_t local_idx = 0;
                 local_idx < autorefine_result.sampled_fragment_triangles.size();
                 ++local_idx) {
                const cgal_autorefine_demo::Triangle& tri =
                    autorefine_result.sampled_fragment_triangles[local_idx];
                const std::size_t tri_idx = autorefine_result.sampled_fragment_indices[local_idx];
                const int assigned_tet = autorefine_result.sampled_fragment_tet_ids(local_idx);
                const int source_sample = autorefine_result.sampled_fragment_source_ids[local_idx];
                std::cout << "\nSample triangle piece " << local_idx << " (from test triangle "
                          << source_sample << ") corresponds to refined triangle " << tri_idx
                          << " [vertices " << tri[0] << ", " << tri[1] << ", " << tri[2] << "]"
                          << std::endl;
                std::cout << "  Assigned tet_id: " << assigned_tet << std::endl;
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t v_id = tri[corner];
                    sampled_vertex_ids.insert(v_id);
                    const cgal_autorefine_demo::RationalPoint& p =
                        autorefine_result.refined_points[v_id];
                    Vector3r original_pos = Vector3r::Zero();
                    bool found_original = false;
                    if (v_id < autorefine_result.original_points.size()) {
                        const cgal_autorefine_demo::RationalPoint& orig_p =
                            autorefine_result.original_points[v_id];
                        original_pos(0) = wmtk::Rational(orig_p.x(), false);
                        original_pos(1) = wmtk::Rational(orig_p.y(), false);
                        original_pos(2) = wmtk::Rational(orig_p.z(), false);
                        found_original = true;
                    } else {
                        for (const auto& sv : autorefine_result.sampled_vertices) {
                            if (sv.point_index == v_id) {
                                original_pos = sv.position;
                                found_original = true;
                                break;
                            }
                        }
                    }
                    Vector3r refined_pos;
                    refined_pos(0) = wmtk::Rational(p.x(), false);
                    refined_pos(1) = wmtk::Rational(p.y(), false);
                    refined_pos(2) = wmtk::Rational(p.z(), false);
                    std::cout << "    Vertex " << v_id << ":" << std::endl;
                    if (found_original) {
                        std::cout << "      Original position: (" << original_pos(0).to_double()
                                  << ", " << original_pos(1).to_double() << ", "
                                  << original_pos(2).to_double() << ")" << std::endl;
                    }
                    std::cout << "      Refined position: (" << refined_pos(0).to_double() << ", "
                              << refined_pos(1).to_double() << ", " << refined_pos(2).to_double()
                              << ")" << std::endl;
                    if (found_original) {
                        Vector3r change = refined_pos - original_pos;
                        std::cout << "      Position change: (" << change(0).to_double() << ", "
                                  << change(1).to_double() << ", " << change(2).to_double() << ")"
                                  << std::endl;
                    }
                    if (v_id < autorefine_result.vertex_tet_sets.size()) {
                        const auto& tet_set = autorefine_result.vertex_tet_sets[v_id];
                        std::cout << "      Shared with tets: ";
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
                        }
                        std::cout << std::endl;
                    }
                }
            }
            std::cout << "\n=== Unique Vertices in Refined Sampled Triangles ===" << std::endl;
            std::cout << "Total unique vertices: " << sampled_vertex_ids.size() << std::endl;
            for (std::size_t v_id : sampled_vertex_ids) {
                const cgal_autorefine_demo::RationalPoint& p =
                    autorefine_result.refined_points[v_id];
                std::cout << "  Vertex " << v_id << ": (" << CGAL::to_double(p.x()) << ", "
                          << CGAL::to_double(p.y()) << ", " << CGAL::to_double(p.z()) << ")"
                          << std::endl;
            }
        } else {
            std::cout << "\nNo refined triangles mapped back to the sampled triangle." << std::endl;
        }
    }
    if (save_debug_meshes == true) {
        if (!autorefine_result.sampled_fragment_triangles.empty()) {
            Eigen::MatrixXi F_refined_sampled(
                autorefine_result.sampled_fragment_triangles.size(),
                3);
            for (std::size_t i = 0; i < autorefine_result.sampled_fragment_triangles.size(); ++i) {
                const cgal_autorefine_demo::Triangle& tri =
                    autorefine_result.sampled_fragment_triangles[i];
                F_refined_sampled(i, 0) = static_cast<int>(tri[0]);
                F_refined_sampled(i, 1) = static_cast<int>(tri[1]);
                F_refined_sampled(i, 2) = static_cast<int>(tri[2]);
            }
            Eigen::MatrixXd V_refined_sampled(autorefine_result.refined_points.size(), 3);
            for (std::size_t i = 0; i < autorefine_result.refined_points.size(); ++i) {
                const cgal_autorefine_demo::RationalPoint& p = autorefine_result.refined_points[i];
                V_refined_sampled(i, 0) = CGAL::to_double(p.x());
                V_refined_sampled(i, 1) = CGAL::to_double(p.y());
                V_refined_sampled(i, 2) = CGAL::to_double(p.z());
            }
            Eigen::VectorXi refined_sampled_tet_ids(
                autorefine_result.sampled_fragment_tet_ids.size());
            for (Eigen::Index i = 0; i < autorefine_result.sampled_fragment_tet_ids.size(); ++i) {
                refined_sampled_tet_ids(i) = autorefine_result.sampled_fragment_tet_ids(i);
            }
            std::string refined_sampled_filename =
                "refined_sampled_triangles_op" + std::to_string(operation_id) + ".vtu";
            vtu_utils::write_triangle_mesh_to_vtu(
                V_refined_sampled,
                F_refined_sampled,
                refined_sampled_filename,
                &refined_sampled_tet_ids,
                "tet_id");
            std::cout << "\nSaved refined sampled triangles to: " << refined_sampled_filename
                      << std::endl;
        }
        Eigen::MatrixXd V_before_double(V_before.rows(), V_before.cols());
        for (int i = 0; i < V_before.rows(); i++) {
            for (int j = 0; j < V_before.cols(); j++) {
                V_before_double(i, j) = V_before(i, j).to_double();
            }
        }
        std::string t_before_filename = "T_before_op" + std::to_string(operation_id) + ".vtu";
        vtu_utils::write_tet_mesh_to_vtu(V_before_double, T_before, t_before_filename);
        std::cout << "Saved V_before and T_before to: " << t_before_filename << std::endl;
    }
    if (!autorefine_result.sampled_fragment_triangles.empty()) {
        std::cout << "\n=== Updating query_surface with refined sampled triangles ===" << std::endl;
        std::map<std::size_t, std::size_t> original_point_to_surface_point;
        for (std::size_t i = 0;
             i < autorefine_result.sampled_vertices.size() && i < unique_point_indices.size();
             ++i) {
            const auto& sv = autorefine_result.sampled_vertices[i];
            std::size_t orig_point_idx = sv.point_index;
            int surface_point_idx = unique_point_indices[static_cast<int>(i)];
            original_point_to_surface_point[orig_point_idx] =
                static_cast<std::size_t>(surface_point_idx);
        }
        std::map<std::size_t, std::size_t> refined_point_to_surface_point;
        std::size_t num_new_points_added = 0;
        for (const auto& [orig_idx, surf_idx] : original_point_to_surface_point) {
            refined_point_to_surface_point[orig_idx] = surf_idx;
        }
        std::set<std::size_t> refined_vertex_ids_used;
        for (const auto& tri : autorefine_result.sampled_fragment_triangles) {
            refined_vertex_ids_used.insert(tri[0]);
            refined_vertex_ids_used.insert(tri[1]);
            refined_vertex_ids_used.insert(tri[2]);
        }
        auto barycentric_total_time = std::chrono::milliseconds(0);
        // Track new points and their original (unrounded) barycentric coordinates for fallback
        std::set<std::size_t> new_point_indices;
        std::map<std::size_t, Vector4r> new_point_original_bc;
        for (std::size_t refined_v_id : refined_vertex_ids_used) {
            if (refined_point_to_surface_point.find(refined_v_id) !=
                refined_point_to_surface_point.end()) {
                continue;
            }
            std::cout << "  Processing new point " << refined_v_id << " (refined index)"
                      << std::endl;
            const cgal_autorefine_demo::RationalPoint& p =
                autorefine_result.refined_points[refined_v_id];
            Vector3r point_pos;
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
                    if (tri[0] == refined_v_id || tri[1] == refined_v_id ||
                        tri[2] == refined_v_id) {
                        local_tet_id = autorefine_result.sampled_fragment_tet_ids(tri_idx);
                        break;
                    }
                }
            }
            if (local_tet_id == -1 || local_tet_id >= T_before.rows()) {
                std::cerr << "Warning: Could not find valid tet_id for new point " << refined_v_id
                          << std::endl;
                continue;
            }
            std::cout << "    Found local_tet_id: " << local_tet_id << std::endl;
            int64_t global_tet_id = -1;
            if (local_tet_id >= 0 && local_tet_id < static_cast<int>(id_map_before.size())) {
                global_tet_id = id_map_before[static_cast<std::size_t>(local_tet_id)];
            }
            if (global_tet_id == -1) {
                std::cerr << "Warning: Could not map local_tet_id " << local_tet_id
                          << " to global_tet_id" << std::endl;
                continue;
            }
            std::cout << "    Mapped to global_tet_id: " << global_tet_id << std::endl;
            Eigen::Vector4i local_tv_ids = T_before.row(local_tet_id);
            Eigen::Vector4i global_tv_ids;
            for (int i = 0; i < 4; ++i) {
                int local_v_id = local_tv_ids(i);
                if (local_v_id >= 0 && local_v_id < static_cast<int>(v_id_map_before.size())) {
                    global_tv_ids(i) =
                        static_cast<int>(v_id_map_before[static_cast<std::size_t>(local_v_id)]);
                } else {
                    std::cerr << "Warning: Invalid local vertex id " << local_v_id << " in tet "
                              << local_tet_id << std::endl;
                    global_tv_ids(i) = -1;
                }
            }
            std::cout << "    Tet vertex ids (local->global): [" << local_tv_ids(0) << ","
                      << local_tv_ids(1) << "," << local_tv_ids(2) << "," << local_tv_ids(3)
                      << "] -> [" << global_tv_ids(0) << "," << global_tv_ids(1) << ","
                      << global_tv_ids(2) << "," << global_tv_ids(3) << "]" << std::endl;
            Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices;
            for (int i = 0; i < 4; ++i) {
                int local_v_id = local_tv_ids(i);
                if (local_v_id >= 0 && local_v_id < V_before.rows()) {
                    tet_vertices.row(i) = V_before.row(local_v_id);
                } else {
                    std::cerr << "Warning: Invalid local vertex id " << local_v_id << std::endl;
                    tet_vertices.row(i).setZero();
                }
            }
            auto barycentric_start = std::chrono::high_resolution_clock::now();
            Vector4r barycentric_coords =
                world_to_barycentric_tet<wmtk::Rational>(point_pos, tet_vertices);
            auto barycentric_end = std::chrono::high_resolution_clock::now();
            barycentric_total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                barycentric_end - barycentric_start);
            std::cout << "    Computed barycentric: [" << barycentric_coords(0).to_double() << ","
                      << barycentric_coords(1).to_double() << ","
                      << barycentric_coords(2).to_double() << ","
                      << barycentric_coords(3).to_double() << "]" << std::endl;
            query_point_tet_r new_point;
            new_point.t_id = global_tet_id;
            std::size_t new_surface_point_idx = surface.points.size();
            // Record original barycentric coords before rounding for potential fallback
            new_point_original_bc[new_surface_point_idx] = barycentric_coords;
            if (do_rounding) {
                new_point.bc = rounding_bc(barycentric_coords);
            } else {
                new_point.bc = barycentric_coords;
            }
            new_point.tv_ids = global_tv_ids;
            surface.points.push_back(new_point);
            new_point_indices.insert(new_surface_point_idx);
            refined_point_to_surface_point[refined_v_id] = new_surface_point_idx;
            num_new_points_added++;
            std::cout << "    Added as surface.points[" << new_surface_point_idx << "]"
                      << std::endl;
        }
        // Check for duplicate points and revert new points to original values if duplicates found
        if (do_rounding) {
            std::cout << "\n=== Checking for duplicate points after rounding ===" << std::endl;
            bool found_duplicate_with_new_point = false;
            for (size_t i = 0; i < surface.points.size(); ++i) {
                const auto& p1 = surface.points[i];
                for (size_t j = i + 1; j < surface.points.size(); ++j) {
                    const auto& p2 = surface.points[j];
                    if (p1.t_id == p2.t_id) {
                        bool bc_equal = true;
                        for (int k = 0; k < 4; ++k) {
                            if (p1.bc(k) != p2.bc(k)) {
                                bc_equal = false;
                                break;
                            }
                        }
                        if (bc_equal) {
                            bool i_is_new = new_point_indices.count(i) > 0;
                            bool j_is_new = new_point_indices.count(j) > 0;
                            if (i_is_new || j_is_new) {
                                found_duplicate_with_new_point = true;
                                std::cout << "  Found duplicate: point " << i << " and point " << j
                                          << " (i_is_new=" << i_is_new << ", j_is_new=" << j_is_new
                                          << ")" << std::endl;
                                // Revert new points to original values
                                if (i_is_new) {
                                    std::cout << "    Reverting point " << i
                                              << " to original (unrounded) bc" << std::endl;
                                    surface.points[i].bc = new_point_original_bc[i];
                                }
                                if (j_is_new) {
                                    std::cout << "    Reverting point " << j
                                              << " to original (unrounded) bc" << std::endl;
                                    surface.points[j].bc = new_point_original_bc[j];
                                }
                            }
                        }
                    }
                }
            }
            if (!found_duplicate_with_new_point) {
                std::cout << "  No duplicate points involving new points found" << std::endl;
            }
            std::cout << "=== End of duplicate check ===" << std::endl;
        }
        std::cout << "  Added " << num_new_points_added << " new points to surface.points"
                  << std::endl;
        std::cout << "Barycentric coordinate computation took " << barycentric_total_time.count()
                  << " ms" << std::endl;
        std::cout << "  Removing " << face_ids.size() << " old triangles that were refined..."
                  << std::endl;
        std::sort(face_ids.begin(), face_ids.end(), std::greater<int>());
        // DEBUG: print the sampled triangles
        if (verbose) {
            for (size_t i = 0; i < face_ids.size(); ++i) {
                int face_id = face_ids[i];
                if (face_id >= 0 && face_id < static_cast<int>(surface.query_triangles.size())) {
                    const auto& tri = surface.query_triangles[face_id];
                    std::cout << "face_ids[" << i << "] = " << face_id
                              << ", triangle = " << tri.transpose() << std::endl;
                } else {
                    std::cout << "face_ids[" << i << "] = " << face_id << " (invalid index)"
                              << std::endl;
                }
            }
        }
        // DEBUG: print the refined_point_to_surface_point mapping
        if (verbose) {
            std::cout << "refined_point_to_surface_point mapping:" << std::endl;
            for (const auto& kv : refined_point_to_surface_point) {
                std::cout << "  refined_v_id " << kv.first << " -> surface_point_idx " << kv.second
                          << std::endl;
            }
        }
        for (int face_id : face_ids) {
            if (face_id >= 0 && face_id < static_cast<int>(surface.query_triangles.size())) {
                surface.query_triangles.erase(surface.query_triangles.begin() + face_id);
                if (face_id < static_cast<int>(surface.tet_ids.size())) {
                    surface.tet_ids.erase(surface.tet_ids.begin() + face_id);
                }
            }
        }
        std::cout << "  Adding " << autorefine_result.sampled_fragment_triangles.size()
                  << " new refined triangles..." << std::endl;
        size_t start_tri_idx = surface.query_triangles.size();
        for (std::size_t i = 0; i < autorefine_result.sampled_fragment_triangles.size(); ++i) {
            const cgal_autorefine_demo::Triangle& refined_tri =
                autorefine_result.sampled_fragment_triangles[i];
            Eigen::Vector3i new_tri;
            bool all_mapped = true;
            for (int corner = 0; corner < 3; ++corner) {
                std::size_t refined_v_id = refined_tri[corner];
                auto it = refined_point_to_surface_point.find(refined_v_id);
                if (it != refined_point_to_surface_point.end()) {
                    new_tri(corner) = static_cast<int>(it->second);
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
            surface.query_triangles.push_back(new_tri);
            int local_tet_id = autorefine_result.sampled_fragment_tet_ids(i);
            int64_t global_tet_id = -1;
            if (local_tet_id >= 0 && local_tet_id < static_cast<int>(id_map_before.size())) {
                global_tet_id = id_map_before[static_cast<std::size_t>(local_tet_id)];
            }
            surface.tet_ids.push_back(static_cast<int>(global_tet_id));
            std::cout << "    Added triangle [" << new_tri(0) << "," << new_tri(1) << ","
                      << new_tri(2) << "] with tet_id " << global_tet_id << std::endl;
        }
        std::cout << "=== Surface update completed ===" << std::endl;
        std::cout << "  Final surface: " << surface.points.size() << " points, "
                  << surface.query_triangles.size() << " triangles" << std::endl;
        // Self-intersection check per tet: revert new points if self-intersection found
        if (do_rounding && !new_point_indices.empty()) {
            std::cout << "\n=== Checking for self-intersection per tet after rounding ==="
                      << std::endl;
            // Find tets that contain new points
            std::set<int64_t> tets_with_new_points;
            for (size_t p_idx : new_point_indices) {
                tets_with_new_points.insert(surface.points[p_idx].t_id);
            }
            std::cout << "  Found " << tets_with_new_points.size()
                      << " tet(s) containing new points" << std::endl;
            // For each tet with new points, check for self-intersection
            for (int64_t tet_id : tets_with_new_points) {
                // Collect all triangles in this tet
                std::vector<size_t> tet_tri_indices;
                for (size_t tri_idx = 0; tri_idx < surface.query_triangles.size(); ++tri_idx) {
                    if (tri_idx < surface.tet_ids.size() && surface.tet_ids[tri_idx] == tet_id) {
                        tet_tri_indices.push_back(tri_idx);
                    }
                }
                if (tet_tri_indices.empty()) {
                    continue;
                }
                // Collect unique points used by these triangles
                std::set<int> tet_points;
                for (size_t tri_idx : tet_tri_indices) {
                    const auto& tri = surface.query_triangles[tri_idx];
                    tet_points.insert(tri(0));
                    tet_points.insert(tri(1));
                    tet_points.insert(tri(2));
                }
                // Build CGAL triangle soup with rational coordinates
                std::map<int, std::size_t> point_to_cgal_idx;
                std::vector<RationalPoint> cgal_points;
                cgal_points.reserve(tet_points.size());
                for (int p_idx : tet_points) {
                    point_to_cgal_idx[p_idx] = cgal_points.size();
                    const auto& qp = surface.points[p_idx];
                    RationalKernel::FT x = rational_to_gmpq(qp.bc(0));
                    RationalKernel::FT y = rational_to_gmpq(qp.bc(1));
                    RationalKernel::FT z = rational_to_gmpq(qp.bc(2));
                    cgal_points.emplace_back(x, y, z);
                }
                std::vector<Triangle> cgal_triangles;
                cgal_triangles.reserve(tet_tri_indices.size());
                for (size_t tri_idx : tet_tri_indices) {
                    const auto& tri = surface.query_triangles[tri_idx];
                    cgal_triangles.push_back(Triangle{
                        point_to_cgal_idx[tri(0)],
                        point_to_cgal_idx[tri(1)],
                        point_to_cgal_idx[tri(2)]});
                }
                // Check for self-intersection
                bool has_self_intersection =
                    PMP::does_triangle_soup_self_intersect(cgal_points, cgal_triangles);
                if (has_self_intersection) {
                    std::cout << "  WARNING: Self-intersection detected in tet_id " << tet_id
                              << ". Reverting new points to original values." << std::endl;
                    // Revert new points in this tet to original values
                    for (int p_idx : tet_points) {
                        if (new_point_indices.count(p_idx) > 0) {
                            std::cout << "    Reverting point " << p_idx
                                      << " to original (unrounded) bc" << std::endl;
                            surface.points[p_idx].bc = new_point_original_bc[p_idx];
                        }
                    }
                } else {
                    std::cout << "  Tet_id " << tet_id << ": self-intersection check passed"
                              << std::endl;
                }
            }
            std::cout << "=== End of self-intersection check ===" << std::endl;
        }
        { // Check manifold property before simplification
            std::cout << "\n=== Checking surface manifold property before simplification ==="
                      << std::endl;
            bool is_manifold = check_surface_manifold_property(surface.query_triangles);
            std::cout << "  Surface is " << (is_manifold ? "manifold" : "NOT manifold")
                      << std::endl;
            if (!is_manifold) {
                throw std::runtime_error(
                    "Surface is not manifold before simplification. Cannot proceed.");
            }
            // Check for unreferenced points
            std::set<int> referenced_points;
            for (const auto& tri : surface.query_triangles) {
                referenced_points.insert(tri(0));
                referenced_points.insert(tri(1));
                referenced_points.insert(tri(2));
            }
            std::vector<int> unreferenced_points;
            for (size_t i = 0; i < surface.points.size(); ++i) {
                if (referenced_points.find(static_cast<int>(i)) == referenced_points.end()) {
                    unreferenced_points.push_back(static_cast<int>(i));
                }
            }
            if (!unreferenced_points.empty()) {
                std::cout << "  WARNING: Found " << unreferenced_points.size()
                          << " unreferenced point(s):" << std::endl;
                for (int p_idx : unreferenced_points) {
                    const auto& qp = surface.points[p_idx];
                    std::cout << "    Point ID: " << p_idx << ", t_id: " << qp.t_id << ", bc: [";
                    for (int i = 0; i < 4; ++i) {
                        std::cout << qp.bc(i).to_double();
                        if (i < 3) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "], tv_ids: [" << qp.tv_ids(0) << ", " << qp.tv_ids(1) << ", "
                              << qp.tv_ids(2) << ", " << qp.tv_ids(3) << "]" << std::endl;
                }
            } else {
                std::cout << "  All points are referenced by triangles" << std::endl;
            }
            std::cout << "=== End of pre-simplification checks ===" << std::endl;
            // Remove unreferenced points before simplification
            if (!unreferenced_points.empty()) {
                std::cout << "\n=== Removing unreferenced points before simplification ==="
                          << std::endl;
                std::cout << "  Removing " << unreferenced_points.size() << " unreferenced point(s)"
                          << std::endl;
                std::cout << "  Removed point IDs: [";
                bool first = true;
                for (int p_idx : unreferenced_points) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << p_idx;
                    first = false;
                }
                std::cout << "]" << std::endl;
                // Build point mapping
                std::set<int> unreferenced_set(
                    unreferenced_points.begin(),
                    unreferenced_points.end());
                std::vector<int> old_to_new_point_map(surface.points.size(), -1);
                int new_point_idx = 0;
                for (size_t i = 0; i < surface.points.size(); ++i) {
                    if (unreferenced_set.find(static_cast<int>(i)) == unreferenced_set.end()) {
                        old_to_new_point_map[i] = new_point_idx++;
                    }
                }
                // Build new points vector
                std::vector<query_point_tet_r> new_points;
                new_points.reserve(new_point_idx);
                for (size_t i = 0; i < surface.points.size(); ++i) {
                    if (unreferenced_set.find(static_cast<int>(i)) == unreferenced_set.end()) {
                        new_points.push_back(surface.points[i]);
                    }
                }
                // Update triangles with new point indices
                std::vector<Eigen::Vector3i> new_triangles;
                new_triangles.reserve(surface.query_triangles.size());
                for (const auto& tri : surface.query_triangles) {
                    Eigen::Vector3i new_tri;
                    new_tri(0) = old_to_new_point_map[tri(0)];
                    new_tri(1) = old_to_new_point_map[tri(1)];
                    new_tri(2) = old_to_new_point_map[tri(2)];
                    if (new_tri(0) >= 0 && new_tri(1) >= 0 && new_tri(2) >= 0) {
                        new_triangles.push_back(new_tri);
                    }
                }
                // Update surface
                surface.points = std::move(new_points);
                surface.query_triangles = std::move(new_triangles);
                std::cout << "  After removing unreferenced points: " << surface.points.size()
                          << " points, " << surface.query_triangles.size() << " triangles"
                          << std::endl;
                std::cout << "=== End of removing unreferenced points ===" << std::endl;
            }
        }

        if (do_simplify) {
            simplify_refined_triangles_by_tet(
                surface,
                start_tri_idx,
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                operation_id);
        }
    }
}

void handle_local_mapping_operation(
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const MatrixXr& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_surface_tet_with_connectivity& surface,
    int operation_id,
    bool do_rounding,
    bool do_simplify,
    bool only_do_arrangement_once)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Handling Local Mapping operation for surface with connectivity" << std::endl;
    auto step1_start = std::chrono::high_resolution_clock::now();
    std::cout << "Mapping all points in the surface to the new connectivity" << std::endl;
    tet_point_tracking::handle_local_mapping_tet_exact(
        V_before,
        T_before,
        id_map_before,
        v_id_map_before,
        V_after,
        T_after,
        id_map_after,
        v_id_map_after,
        surface.points,
        false);
    auto step1_end = std::chrono::high_resolution_clock::now();
    auto step1_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(step1_end - step1_start);
    std::cout << "Mapping all points in the surface to the new connectivity completed" << std::endl;
    std::cout << "Step1 (point mapping) took " << step1_duration.count() << " ms" << std::endl;
    if (!only_do_arrangement_once) {
        auto step2_start = std::chrono::high_resolution_clock::now();
        bool save_debug_meshes = false;
        bool verbose = false;
        surface_triangle_arrangement(
            V_before,
            T_before,
            id_map_before,
            v_id_map_before,
            id_map_after,
            surface,
            operation_id,
            do_rounding,
            verbose,
            save_debug_meshes,
            do_simplify);
        auto step2_end = std::chrono::high_resolution_clock::now();
        auto step2_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(step2_end - step2_start);
        std::cout << "Step2 (surface triangle arrangement) took " << step2_duration.count() << " ms"
                  << std::endl;
    } else {
        std::cout << "Skipping per-operation arrangement (only_do_arrangement_once=true)"
                  << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "handle_local_mapping_operation took " << duration.count() << " ms" << std::endl;
}

void track_one_operation(
    const nlohmann::json& operation_log,
    query_surface_tet_with_connectivity& surface,
    bool do_forward,
    int operation_id,
    bool do_rounding,
    bool do_simplify,
    bool only_do_arrangement_once)
{
    std::string operation_name = operation_log["operation_name"];
    std::cout << "Tracking operation: " << operation_name << " (ID: " << operation_id << ")"
              << std::endl;
    Eigen::MatrixXi T_after, T_before;
    std::vector<int64_t> id_map_after, id_map_before;
    bool has_tet_context = false;
    if (operation_name == "MeshConsolidate") {
        std::cout << "  This operation is Consolidate" << std::endl;
        std::vector<int64_t> tet_ids_maps;
        std::vector<int64_t> vertex_ids_maps;
        parse_consolidate_file_tet(operation_log, tet_ids_maps, vertex_ids_maps);
        handle_consolidate_operation(tet_ids_maps, vertex_ids_maps, surface, do_forward);
    } else {
        std::cout << "  This operation is " << operation_name << std::endl;
        Eigen::MatrixXd V_after_double, V_before_double;
        std::vector<int64_t> v_id_map_after, v_id_map_before;
        parse_non_collapse_file_tet(
            operation_log,
            V_before_double,
            T_before,
            id_map_before,
            v_id_map_before,
            V_after_double,
            T_after,
            id_map_after,
            v_id_map_after,
            operation_id);
        has_tet_context = true;
        MatrixXr V_before(V_before_double.rows(), V_before_double.cols());
        MatrixXr V_after(V_after_double.rows(), V_after_double.cols());
        for (int i = 0; i < V_before_double.rows(); i++) {
            for (int j = 0; j < V_before_double.cols(); j++) {
                V_before(i, j) = wmtk::Rational(V_before_double(i, j));
            }
        }
        for (int i = 0; i < V_after_double.rows(); i++) {
            for (int j = 0; j < V_after_double.cols(); j++) {
                V_after(i, j) = wmtk::Rational(V_after_double(i, j));
            }
        }
        if (do_forward) {
            handle_local_mapping_operation(
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                surface,
                operation_id,
                do_rounding,
                do_simplify,
                only_do_arrangement_once);
        } else {
            handle_local_mapping_operation(
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                surface,
                operation_id,
                do_rounding,
                do_simplify,
                only_do_arrangement_once);
        }
    }
    std::cout << "  Operation " << operation_id << " completed" << std::endl;

    if (has_tet_context) {
        // const auto& sanity_T = do_forward ? T_after : T_before;
        const auto& sanity_id_map = do_forward ? id_map_after : id_map_before;
        sanity_check_triangles(surface, sanity_id_map);
        std::cout << "  Triangle sanity check completed" << std::endl;
    } else {
        std::cout << "  Skipping triangle sanity check (no tet context for consolidate)"
                  << std::endl;
    }

    {
        bool is_manifold = check_surface_manifold_property(surface.query_triangles);
        if (is_manifold) {
            std::cout << "Surface is manifold" << std::endl;
        } else {
            std::cout << "Surface is not manifold" << std::endl;
            std::cout << "[" << std::endl;
            for (size_t i = 0; i < surface.query_triangles.size(); ++i) {
                const auto& tri = surface.query_triangles[i];
                std::cout << "[" << tri(0) << ", " << tri(1) << ", " << tri(2) << "]";
                if (i != surface.query_triangles.size() - 1) {
                    std::cout << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "]" << std::endl;
            throw std::runtime_error("Error: surface is not manifold");
        }
    }
}

void track_all_operations(
    const std::filesystem::path& dirPath,
    query_surface_tet_with_connectivity& surface,
    bool do_forward,
    bool do_rounding)
{
    std::cout << "Tracking all operations from directory: " << dirPath << std::endl;
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
        nlohmann::json operation_log = reader.get_operation(operation_index);
        if (operation_log.empty()) {
            std::cerr << "Failed to read operation " << operation_index << std::endl;
            continue;
        }
        std::cout << "\n=== Processing operation " << (i + 1) << "/" << total_ops
                  << " (index: " << operation_index << ") ===" << std::endl;
        track_one_operation(
            operation_log,
            surface,
            do_forward,
            static_cast<int>(operation_index),
            do_rounding);
    }
    std::cout << "\n=== All operations completed ===" << std::endl;
    std::cout << "Final surface state: " << surface.points.size() << " points, "
              << surface.query_triangles.size() << " triangles" << std::endl;
}

Vector4r rounding_bc(const Vector4r& bc)
{
    Vector4r result;
    for (int i = 0; i < 4; i++) {
        result(i) = wmtk::Rational(bc(i).to_double());
    }
    int target_idx = -1;
    for (int i = 0; i < 4; i++) {
        double val = result(i).to_double();
        if (val != 0.0 && val != 1.0) {
            target_idx = i;
            break;
        }
    }
    if (target_idx != -1) {
        wmtk::Rational sum_others = wmtk::Rational(0);
        for (int i = 0; i < 4; i++) {
            if (i != target_idx) {
                sum_others += result(i);
            }
        }
        result(target_idx) = wmtk::Rational(1) - sum_others;
    }
    return result;
}

std::pair<std::vector<int>, std::vector<Vector4r>> get_point_representations(
    const int local_t_id,
    const Vector4r& local_bc,
    const Eigen::MatrixXi& T_local)
{
    std::vector<int> all_possible_t_ids;
    std::vector<Vector4r> all_possible_bcs;
    all_possible_t_ids.push_back(local_t_id);
    all_possible_bcs.push_back(local_bc);
    std::vector<int> non_zeros_vid;
    std::vector<wmtk::Rational> non_zeros_bc;
    for (int i = 0; i < 4; i++) {
        if (local_bc(i) != 0) {
            non_zeros_vid.push_back(T_local(local_t_id, i));
            non_zeros_bc.push_back(local_bc(i));
        }
    }
    if (non_zeros_vid.size() < 4) {
        for (int t_id = 0; t_id < T_local.rows(); t_id++) {
            if (t_id == local_t_id) continue;
            bool contains_all = true;
            Vector4r bc_tmp = Vector4r::Zero();
            for (int i = 0; i < non_zeros_vid.size(); i++) {
                int v_idx = non_zeros_vid[i];
                bool found = false;
                for (int j = 0; j < 4; j++) {
                    if (T_local(t_id, j) == v_idx) {
                        found = true;
                        bc_tmp(j) = non_zeros_bc[i];
                    }
                }
                if (!found) {
                    contains_all = false;
                    break;
                }
            }
            if (contains_all) {
                all_possible_t_ids.push_back(t_id);
                all_possible_bcs.push_back(bc_tmp);
            }
        }
    }
    return {all_possible_t_ids, all_possible_bcs};
}

} // namespace tet_surface_tracking_with_connectivity

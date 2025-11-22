#include "tet_surface_tracking_with_connectivity.hpp"
#include <CGAL/number_utils.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "batch_operation_log_reader.hpp"
#include "cgal_autorefine_utils_rational.hpp"
#include "tet_point_tracking.hpp"
#include "tet_surface_sampling.hpp"
#include "tet_track_operations.hpp"
#include "tet_track_operations_internal.hpp"
#include "vtu_utils.hpp"

namespace tet_surface_tracking_with_connectivity {

std::pair<MatrixXr, Eigen::MatrixXi> surface_to_world_positions_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V)
{
    std::cout << "Converting query surface with connectivity to world positions..." << std::endl;

    // Allocate output matrices
    // V_out: one row per unique point (not 3 per triangle)
    MatrixXr V_out(query_surface.points.size(), 3);
    Eigen::MatrixXi F_out(query_surface.query_triangles.size(), 3);

    // Convert each point to world coordinates
    for (size_t i = 0; i < query_surface.points.size(); i++) {
        const auto& pt = query_surface.points[i];

        // Get vertices of the tetrahedron
        Eigen::Vector4i tet_verts = pt.tv_ids;

        // Calculate real position using barycentric coordinates
        Vector3r world_pos = Vector3r::Zero();
        for (int j = 0; j < 4; j++) {
            world_pos += pt.bc(j) * V.row(tet_verts(j)).transpose();
        }
        V_out.row(i) = world_pos.transpose();
    }

    // Copy triangle connectivity
    for (size_t i = 0; i < query_surface.query_triangles.size(); i++) {
        F_out.row(i) = query_surface.query_triangles[i];
    }

    std::cout << "Converted " << V_out.rows() << " vertices and " << F_out.rows() << " triangles"
              << std::endl;

    return {V_out, F_out};
}

void write_surface_to_vtu_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V,
    const std::string& filename)
{
    std::cout << "Writing surface with connectivity to VTU file: " << filename << std::endl;

    // First convert to world positions
    auto [V_rational, F] = surface_to_world_positions_rational(query_surface, V);

    // Convert rational coordinates to double for VTU output
    Eigen::MatrixXd V_double(V_rational.rows(), V_rational.cols());
    for (int i = 0; i < V_rational.rows(); i++) {
        for (int j = 0; j < V_rational.cols(); j++) {
            V_double(i, j) = V_rational(i, j).to_double();
        }
    }

    // Prepare tet_id as cell scalar (one value per triangle)
    Eigen::VectorXi tet_ids_vector(query_surface.tet_ids.size());
    for (size_t i = 0; i < query_surface.tet_ids.size(); i++) {
        tet_ids_vector(i) = query_surface.tet_ids[i];
    }

    // Write to VTU using existing utility function
    vtu_utils::write_triangle_mesh_to_vtu(V_double, F, filename, &tet_ids_vector, "tet_id");

    std::cout << "Successfully wrote " << V_double.rows() << " vertices and " << F.rows()
              << " triangles to " << filename << std::endl;
    std::cout << "  with tet_id property attached to each triangle" << std::endl;
}

void write_surface_to_vtu(
    const query_surface_tet_with_connectivity& query_surface,
    const Eigen::MatrixXd& V,
    const std::string& filename)
{
    // Convert double matrix to rational matrix
    MatrixXr V_rational(V.rows(), V.cols());
    for (int i = 0; i < V.rows(); i++) {
        for (int j = 0; j < V.cols(); j++) {
            V_rational(i, j) = wmtk::Rational(V(i, j));
        }
    }

    // Call rational version
    write_surface_to_vtu_rational(query_surface, V_rational, filename);
}

void print_surface_area_statistics(const MatrixXr& surface_V, const Eigen::MatrixXi& surface_F)
{
    // TODO: Implement triangle area statistics calculation and printing
    std::cout << "\n=== Surface Area Statistics ===" << std::endl;
    std::cout << "Number of triangles: " << surface_F.rows() << std::endl;
    std::cout << "================================\n" << std::endl;
}

void check_surface_manifold_property(const MatrixXr& surface_V, const Eigen::MatrixXi& surface_F)
{
    // TODO: Implement manifold property checking
    std::cout << "Checking manifold property..." << std::endl;
}

void run_backward_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold)
{
    std::cout << "Backward tracking surface with connectivity" << std::endl;

    // Step 1: Read or sample the query surface with connectivity
    std::string query_surface_filename = surface_file.string();
    query_surface_tet_with_connectivity query_surface;
    if (!std::filesystem::exists(query_surface_filename)) {
        std::cout << "query_surface not found, sampling and writing to file..." << std::endl;
        // User must provide surface file or use external sampling functions
        // query_surface =
        // tet_surface_sampling::sample_query_surface_tet_with_connectivity(T_after, V_after);
        int N = 5; // number of slicing planes (can be parameterized)
        int axis = 2; // slice axis (can be parameterized)
        double min_coord = V_after.col(axis).minCoeff();
        double max_coord = V_after.col(axis).maxCoeff();
        double plane_coord =
            min_coord + (max_coord - min_coord) * (N + 1) / (N + 2); // single point near last slice
        query_surface = tet_surface_sampling::slice_tet_mesh_with_axis_plane(
            T_after,
            V_after,
            axis,
            plane_coord);
        write_surface_connectivity_to_file(query_surface, query_surface_filename);
    } else {
        std::cout << "query_surface found, reading from file..." << std::endl;
        query_surface = read_surface_connectivity_from_file(query_surface_filename);
    }

    std::string model_name = "model";
    auto dir_str = operation_logs_dir.filename().string();
    const std::string prefix = "operation_log_";
    if (dir_str.size() > prefix.size() && dir_str.substr(0, prefix.size()) == prefix) {
        model_name = dir_str.substr(prefix.size());
    }
    write_surface_to_vtu(
        query_surface,
        V_after,
        model_name + "_query_surface_tet_with_connectivity_after.vtu");

    {
        // DEBUG: sanitity check for the input query_surface
        for (int i = 0; i < query_surface.query_triangles.size(); i++) {
            const Eigen::Vector3i& tri = query_surface.query_triangles[i];
            int tri_tet_id = query_surface.tet_ids[i];

            const Eigen::Vector4i& relevant_vids = T_after.row(tri_tet_id);

            // check points
            for (int j = 0; j < 3; j++) {
                const auto& pt = query_surface.points[tri[j]];
                if (pt.t_id != tri_tet_id) {
                    std::cout << "i: " << i << ", tri_tet_id: " << tri_tet_id
                              << ", pt: [t_id=" << pt.t_id << ", bc=(" << pt.bc[0].to_double()
                              << ", " << pt.bc[1].to_double() << ", " << pt.bc[2].to_double()
                              << ", " << pt.bc[3].to_double() << ")]" << std::endl;

                    const auto tv_ids = pt.tv_ids;
                    for (int bc_idx = 0; bc_idx < 4; bc_idx++) {
                        if (pt.bc(bc_idx) != 0) {
                            int v_idx = tv_ids(bc_idx);
                            if (std::find(relevant_vids.data(), relevant_vids.data() + 4, v_idx) ==
                                relevant_vids.data() + 4) {
                                std::cout << "ERROR: " << "v_idx: " << v_idx
                                          << " is not in relevant_vids" << std::endl;
                                std::cout << "bc of this point: " << pt.bc(bc_idx).to_double()
                                          << std::endl;
                                throw std::runtime_error("Error: point not in relevant_vids");
                            }
                        }
                    }
                }
            }
        }
    }

    // step2 do the backward tracking
    std::cout << "Doing backward tracking..." << std::endl;
    track_all_operations(operation_logs_dir, query_surface, false);
    std::cout << "Backward tracking completed" << std::endl;

    // step3 write the surface to file
    write_surface_connectivity_to_file(
        query_surface,
        model_name + "_query_surface_tet_with_connectivity_before.json");
    write_surface_to_vtu(
        query_surface,
        V_before,
        model_name + "_query_surface_tet_with_connectivity_before.vtu");

    // TODO: results sanity check
}

void write_surface_connectivity_to_file(
    const query_surface_tet_with_connectivity& surface,
    const std::string& filename)
{
    std::cout << "Writing surface with connectivity to file: " << filename << std::endl;

    json j;
    j["num_points"] = surface.points.size();
    j["num_triangles"] = surface.query_triangles.size();

    // Write all points
    for (size_t i = 0; i < surface.points.size(); ++i) {
        const auto& pt = surface.points[i];
        json pt_json;
        pt_json["t_id"] = pt.t_id;

        // Store barycentric coordinates as strings to preserve rational precision
        pt_json["bc"] = {
            pt.bc[0].serialize(),
            pt.bc[1].serialize(),
            pt.bc[2].serialize(),
            pt.bc[3].serialize()};

        // Store tetrahedron vertex ids
        pt_json["tv_ids"] = {pt.tv_ids[0], pt.tv_ids[1], pt.tv_ids[2], pt.tv_ids[3]};

        j["points"].push_back(pt_json);
    }

    // Write all triangles
    for (size_t i = 0; i < surface.query_triangles.size(); ++i) {
        const auto& tri = surface.query_triangles[i];
        json tri_json;
        tri_json["indices"] = {tri[0], tri[1], tri[2]};

        if (i < surface.tet_ids.size()) {
            tri_json["tet_id"] = surface.tet_ids[i];
        }

        j["triangles"].push_back(tri_json);
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
        std::cout << "Successfully wrote " << surface.points.size() << " points and "
                  << surface.query_triangles.size() << " triangles to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

query_surface_tet_with_connectivity read_surface_connectivity_from_file(const std::string& filename)
{
    std::cout << "Reading surface with connectivity from file: " << filename << std::endl;
    query_surface_tet_with_connectivity surface;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return surface;
    }

    json j;
    file >> j;
    file.close();

    size_t num_points = j["num_points"];
    size_t num_triangles = j["num_triangles"];

    surface.points.resize(num_points);
    surface.query_triangles.resize(num_triangles);
    surface.tet_ids.resize(num_triangles);

    // Read all points
    for (size_t i = 0; i < num_points; ++i) {
        const auto& pt_json = j["points"][i];
        auto& pt = surface.points[i];

        pt.t_id = pt_json["t_id"];

        // Read barycentric coordinates from serialized strings using deserialize
        const auto& bc_array = pt_json["bc"];
        pt.bc = Eigen::Matrix<wmtk::Rational, 4, 1>(
            wmtk::Rational::deserialize(bc_array[0]),
            wmtk::Rational::deserialize(bc_array[1]),
            wmtk::Rational::deserialize(bc_array[2]),
            wmtk::Rational::deserialize(bc_array[3]));

        // Read tetrahedron vertex ids
        const auto& tv_array = pt_json["tv_ids"];
        pt.tv_ids = Eigen::Vector4i(tv_array[0], tv_array[1], tv_array[2], tv_array[3]);
    }

    // Read all triangles
    for (size_t i = 0; i < num_triangles; ++i) {
        const auto& tri_json = j["triangles"][i];
        auto& tri = surface.query_triangles[i];

        const auto& indices = tri_json["indices"];
        tri = Eigen::Vector3i(indices[0], indices[1], indices[2]);

        if (tri_json.contains("tet_id")) {
            surface.tet_ids[i] = tri_json["tet_id"];
        }
    }

    std::cout << "Successfully read " << surface.points.size() << " points and "
              << surface.query_triangles.size() << " triangles from " << filename << std::endl;

    return surface;
}

void handle_consolidate_operation(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet_with_connectivity& surface,
    bool forward)
{
    std::cout << "Handling Consolidate operation for surface with connectivity" << std::endl;

    // Step 1: Handle points using handle_consolidate_tet
    tet_point_tracking::handle_consolidate_tet<wmtk::Rational>(
        tet_ids_maps,
        vertex_ids_maps,
        surface.points,
        forward);

    // Step 2: Handle tet_ids (treat them like t_id in query_point_tet)
    if (!forward) {
        // Backward: direct mapping
        for (auto& tet_id : surface.tet_ids) {
            if (tet_id >= 0) {
                tet_id = tet_ids_maps[tet_id];
            }
        }
    } else {
        // Forward: search for the old value in the map
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
    bool do_rounding = true,
    bool verbose = false,
    bool save_debug_meshes = false)
{
    // step2: get all faces in surface.triangle that is in id_map_after
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

    // Step 2.1: Collect all unique point indices used by selected triangles
    std::set<int> unique_point_indices_set;
    for (int face_id : face_ids) {
        const Eigen::Vector3i& tri = surface.query_triangles[face_id];
        unique_point_indices_set.insert(tri[0]);
        unique_point_indices_set.insert(tri[1]);
        unique_point_indices_set.insert(tri[2]);
    }

    // Convert set to vector for indexing
    std::vector<int> unique_point_indices(
        unique_point_indices_set.begin(),
        unique_point_indices_set.end());

    // Step 2.2: Create mapping from global point index to local point index
    std::map<int, int> global_to_local_point_map;
    for (int local_idx = 0; local_idx < unique_point_indices.size(); local_idx++) {
        global_to_local_point_map[unique_point_indices[local_idx]] = local_idx;
    }

    // Step 2.3: Build local_triangles_F by remapping triangle indices to local indices
    Eigen::MatrixXi local_triangles_F(face_ids.size(), 3);
    for (int i = 0; i < face_ids.size(); i++) {
        int face_id = face_ids[i];
        const Eigen::Vector3i& global_tri = surface.query_triangles[face_id];

        // Map each vertex from global to local index
        local_triangles_F(i, 0) = global_to_local_point_map[global_tri[0]];
        local_triangles_F(i, 1) = global_to_local_point_map[global_tri[1]];
        local_triangles_F(i, 2) = global_to_local_point_map[global_tri[2]];
    }
    if (verbose) {
        std::cout << "Built local triangles mesh: " << unique_point_indices.size() << " vertices, "
                  << local_triangles_F.rows() << " faces" << std::endl;
    }
    // Step 2.4: Prepare sampled points for autorefine
    std::vector<cgal_autorefine_demo::SampledPointInputRational> sampled_points;
    sampled_points.reserve(unique_point_indices.size());

    for (int local_idx = 0; local_idx < unique_point_indices.size(); local_idx++) {
        int global_idx = unique_point_indices[local_idx];
        const auto& pt = surface.points[global_idx];

        cgal_autorefine_demo::SampledPointInputRational sampled_pt;
        sampled_pt.tet_index = -1;
        // Find the position of pt.t_id in id_map_before
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

                // find the alternative representation for this point in this local patch by
                // find all the non-zero bcs and vids in the local patch
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

        // TODO: implement the case that the point has multiple representations in the local
        // patch

        sampled_points.push_back(sampled_pt);
    }

    if (verbose) {
        // Print sampled points
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
        // Print local_triangles_F
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
    // Step 2.5: Call autorefine_sampled_triangles_rational on V_before and T_before

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

    // Extract and print refined sampled triangles
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

                // Print each vertex's position change
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t v_id = tri[corner];
                    sampled_vertex_ids.insert(v_id);
                    const cgal_autorefine_demo::RationalPoint& p =
                        autorefine_result.refined_points[v_id];

                    // Get original position from original_points if this vertex existed before
                    // refine
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
                        // Try to find in sampled_vertices (for sampled points added before
                        // refine)
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

                    // Print tet sets for this vertex
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
        // Save refined sampled triangles and V_before, T_before to VTU files
        if (!autorefine_result.sampled_fragment_triangles.empty()) {
            // Convert refined sampled triangles to Eigen matrices
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

            // Convert refined points to Eigen::MatrixXd
            Eigen::MatrixXd V_refined_sampled(autorefine_result.refined_points.size(), 3);
            for (std::size_t i = 0; i < autorefine_result.refined_points.size(); ++i) {
                const cgal_autorefine_demo::RationalPoint& p = autorefine_result.refined_points[i];
                V_refined_sampled(i, 0) = CGAL::to_double(p.x());
                V_refined_sampled(i, 1) = CGAL::to_double(p.y());
                V_refined_sampled(i, 2) = CGAL::to_double(p.z());
            }

            // Prepare tet_id vector for refined sampled triangles
            Eigen::VectorXi refined_sampled_tet_ids(
                autorefine_result.sampled_fragment_tet_ids.size());
            for (Eigen::Index i = 0; i < autorefine_result.sampled_fragment_tet_ids.size(); ++i) {
                refined_sampled_tet_ids(i) = autorefine_result.sampled_fragment_tet_ids(i);
            }

            // Save refined sampled triangles with tet_id
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

        // Convert V_before and T_before to double and save
        Eigen::MatrixXd V_before_double(V_before.rows(), V_before.cols());
        for (int i = 0; i < V_before.rows(); i++) {
            for (int j = 0; j < V_before.cols(); j++) {
                V_before_double(i, j) = V_before(i, j).to_double();
            }
        }

        std::string v_before_filename = "V_before_op" + std::to_string(operation_id) + ".vtu";
        std::string t_before_filename = "T_before_op" + std::to_string(operation_id) + ".vtu";
        vtu_utils::write_tet_mesh_to_vtu(V_before_double, T_before, t_before_filename);
        std::cout << "Saved V_before and T_before to: " << t_before_filename << std::endl;
    }
    // Step 3: Update query_surface with refined sampled triangles
    // ====================================================================
    // This step updates the surface.points and surface.query_triangles based on
    // the refined sampled triangles from autorefine_result.
    //
    // Process:
    // 1. Identify new points added during refine (not in original_points)
    // 2. For each new point:
    //    - Get its position from refined_points
    //    - Find which local_tet_id it belongs to (from vertex_tet_sets or
    //    sampled_fragment_tet_ids)
    //    - Compute barycentric coordinates within that tet
    //    - Map local_tet_id to global_tet_id using id_map_before
    //    - Map tet vertex indices using v_id_map_before
    //    - Add to surface.points
    // 3. Build mapping from refined_points index to surface.points index
    // 4. Update surface.query_triangles using new indices
    // ====================================================================

    if (!autorefine_result.sampled_fragment_triangles.empty()) {
        std::cout << "\n=== Updating query_surface with refined sampled triangles ===" << std::endl;

        // Step 3.1: Build mapping from original_points index to surface.points index
        // -------------------------------------------------------------------------
        // original_points contains: [tet face vertices..., sampled point vertices...]
        // sampled_points were added after tet faces, so their indices start after tet face
        // count We need to map sampled points in original_points to unique_point_indices in
        // surface.points
        std::map<std::size_t, std::size_t> original_point_to_surface_point;

        // Find where sampled points start in original_points
        // sampled_vertices contain the point_index in original_points for each sampled
        // point The sampled points correspond to unique_point_indices in surface.points
        for (std::size_t i = 0;
             i < autorefine_result.sampled_vertices.size() && i < unique_point_indices.size();
             ++i) {
            const auto& sv = autorefine_result.sampled_vertices[i];
            std::size_t orig_point_idx = sv.point_index;
            int surface_point_idx = unique_point_indices[static_cast<int>(i)];
            original_point_to_surface_point[orig_point_idx] =
                static_cast<std::size_t>(surface_point_idx);
        }

        // Step 3.2: Identify new points and add them to surface.points
        // --------------------------------------------------------------
        // New points are those in refined_points that are not in original_points
        // Mapping: refined_points index -> surface.points index
        std::map<std::size_t, std::size_t> refined_point_to_surface_point;
        std::size_t num_new_points_added = 0;

        // First, map existing points (from original_points)
        for (const auto& [orig_idx, surf_idx] : original_point_to_surface_point) {
            refined_point_to_surface_point[orig_idx] = surf_idx;
        }

        // Then, process all vertices used in refined sampled triangles
        std::set<std::size_t> refined_vertex_ids_used;
        for (const auto& tri : autorefine_result.sampled_fragment_triangles) {
            refined_vertex_ids_used.insert(tri[0]);
            refined_vertex_ids_used.insert(tri[1]);
            refined_vertex_ids_used.insert(tri[2]);
        }

        auto barycentric_total_time = std::chrono::milliseconds(0);
        // For each vertex used in refined triangles
        for (std::size_t refined_v_id : refined_vertex_ids_used) {
            // Skip if already mapped (existing point)
            if (refined_point_to_surface_point.find(refined_v_id) !=
                refined_point_to_surface_point.end()) {
                continue;
            }

            // This is a new point added during refine
            std::cout << "  Processing new point " << refined_v_id << " (refined index)"
                      << std::endl;

            // Get the point's position in world coordinates (exact conversion, no precision
            // loss)
            const cgal_autorefine_demo::RationalPoint& p =
                autorefine_result.refined_points[refined_v_id];
            Vector3r point_pos;
            point_pos(0) = wmtk::Rational(p.x(), false);
            point_pos(1) = wmtk::Rational(p.y(), false);
            point_pos(2) = wmtk::Rational(p.z(), false);

            // Step 3.2.1: Find which local_tet_id this point belongs to
            // ---------------------------------------------------------
            // Try to get from vertex_tet_sets first (most reliable)
            int local_tet_id = -1;
            if (refined_v_id < autorefine_result.vertex_tet_sets.size()) {
                const auto& tet_set = autorefine_result.vertex_tet_sets[refined_v_id];
                if (!tet_set.empty()) {
                    // Use the first tet in the set
                    local_tet_id = *tet_set.begin();
                }
            }

            // Fallback: try to get from sampled_fragment_tet_ids
            // Find a triangle that uses this vertex and get its tet_id
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

            // Step 3.2.2: Map local_tet_id to global_tet_id using id_map_before
            // ------------------------------------------------------------------
            // id_map_before maps: local index -> global tet_id
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

            // Step 3.2.3: Get tet vertex indices and map them using v_id_map_before
            // -----------------------------------------------------------------------
            // T_before uses local vertex indices, we need to map to global vertex indices
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

            // Step 3.2.4: Compute barycentric coordinates of the point in the tet
            // --------------------------------------------------------------------
            // Use local T_before and local V_before for calculation
            // Get tet vertices in world coordinates (using V_before with local indices)
            Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices;
            for (int i = 0; i < 4; ++i) {
                int local_v_id = local_tv_ids(i);
                if (local_v_id >= 0 && local_v_id < V_before.rows()) {
                    tet_vertices.row(i) = V_before.row(local_v_id);
                } else {
                    std::cerr << "Warning: Invalid local vertex id " << local_v_id << std::endl;
                    // Use zero as fallback
                    tet_vertices.row(i).setZero();
                }
            }

            // Compute barycentric coordinates using world_to_barycentric_tet with local
            // vertices
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

            // Step 3.2.5: Create new query_point_tet_r and add to surface.points
            // ------------------------------------------------------------------
            query_point_tet_r new_point;
            new_point.t_id = global_tet_id;
            if (do_rounding) {
                new_point.bc[0] = wmtk::Rational(barycentric_coords(0).to_double());
                new_point.bc[1] = wmtk::Rational(barycentric_coords(1).to_double());
                new_point.bc[2] = wmtk::Rational(barycentric_coords(2).to_double());
                new_point.bc[3] = wmtk::Rational(barycentric_coords(3).to_double());
                new_point.bc /= new_point.bc.sum();
            } else {
                new_point.bc = barycentric_coords;
            }
            new_point.tv_ids = global_tv_ids;

            // Add to surface.points and record the mapping
            std::size_t new_surface_point_idx = surface.points.size();
            surface.points.push_back(new_point);
            refined_point_to_surface_point[refined_v_id] = new_surface_point_idx;
            num_new_points_added++;

            std::cout << "    Added as surface.points[" << new_surface_point_idx << "]"
                      << std::endl;
        }

        std::cout << "  Added " << num_new_points_added << " new points to surface.points"
                  << std::endl;
        std::cout << "Barycentric coordinate computation took " << barycentric_total_time.count()
                  << " ms" << std::endl;

        // Step 3.3: Update surface.query_triangles with refined sampled triangles
        // ------------------------------------------------------------------------
        // Remove old triangles that were refined (those in face_ids)
        // and add new refined triangles

        // Step 3.3.1: Remove old triangles that were refined
        // ---------------------------------------------------
        // face_ids contains indices of triangles in surface.query_triangles that were
        // refined We need to remove these triangles (in reverse order to maintain indices)
        std::cout << "  Removing " << face_ids.size() << " old triangles that were refined..."
                  << std::endl;
        std::sort(face_ids.begin(), face_ids.end(), std::greater<int>());
        for (int face_id : face_ids) {
            if (face_id >= 0 && face_id < static_cast<int>(surface.query_triangles.size())) {
                surface.query_triangles.erase(surface.query_triangles.begin() + face_id);
                if (face_id < static_cast<int>(surface.tet_ids.size())) {
                    surface.tet_ids.erase(surface.tet_ids.begin() + face_id);
                }
            }
        }

        // Step 3.3.2: Add new refined triangles
        // --------------------------------------
        // For each refined sampled triangle, create a new triangle in
        // surface.query_triangles
        std::cout << "  Adding " << autorefine_result.sampled_fragment_triangles.size()
                  << " new refined triangles..." << std::endl;
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

            // Add the new triangle
            surface.query_triangles.push_back(new_tri);

            // Add corresponding tet_id
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
    int operation_id)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Handling Local Mapping operation for surface with connectivity" << std::endl;

    // step1:map all points in the surface
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

    // step2: refine and update surface triangles
    auto step2_start = std::chrono::high_resolution_clock::now();
    surface_triangle_arrangement(
        V_before,
        T_before,
        id_map_before,
        v_id_map_before,
        id_map_after,
        surface,
        operation_id);
    auto step2_end = std::chrono::high_resolution_clock::now();
    auto step2_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(step2_end - step2_start);
    std::cout << "Step2 (surface triangle arrangement) took " << step2_duration.count() << " ms"
              << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "handle_local_mapping_operation took " << duration.count() << " ms" << std::endl;
}

void track_one_operation(
    const nlohmann::json& operation_log,
    query_surface_tet_with_connectivity& surface,
    bool do_forward,
    int operation_id)
{
    std::string operation_name = operation_log["operation_name"];
    std::cout << "Tracking operation: " << operation_name << " (ID: " << operation_id << ")"
              << std::endl;

    if (operation_name == "MeshConsolidate") {
        std::cout << "  This operation is Consolidate" << std::endl;
        std::vector<int64_t> tet_ids_maps;
        std::vector<int64_t> vertex_ids_maps;
        parse_consolidate_file_tet(operation_log, tet_ids_maps, vertex_ids_maps);

        handle_consolidate_operation(tet_ids_maps, vertex_ids_maps, surface, do_forward);
    } else {
        std::cout << "  This operation is " << operation_name << std::endl;

        // Parse operation data
        Eigen::MatrixXi T_after, T_before;
        Eigen::MatrixXd V_after_double, V_before_double;
        std::vector<int64_t> id_map_after, id_map_before;
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

        // Convert double matrices to rational matrices
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

        // Call handle_local_mapping_operation with appropriate direction
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
                operation_id);
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
                operation_id);
        }
    }

    std::cout << "  Operation " << operation_id << " completed" << std::endl;
}

void track_all_operations(
    const std::filesystem::path& dirPath,
    query_surface_tet_with_connectivity& surface,
    bool do_forward)
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

    // Iterate through operations in the appropriate order
    for (size_t i = 0; i < total_ops; ++i) {
        size_t operation_index = i;
        if (!do_forward) {
            // Backward tracking: process operations in reverse order
            operation_index = total_ops - 1 - i;
        }

        json operation_log = reader.get_operation(operation_index);
        if (operation_log.empty()) {
            std::cerr << "Failed to read operation " << operation_index << std::endl;
            continue;
        }

        std::cout << "\n=== Processing operation " << (i + 1) << "/" << total_ops
                  << " (index: " << operation_index << ") ===" << std::endl;

        track_one_operation(operation_log, surface, do_forward, static_cast<int>(operation_index));
    }

    std::cout << "\n=== All operations completed ===" << std::endl;
    std::cout << "Final surface state: " << surface.points.size() << " points, "
              << surface.query_triangles.size() << " triangles" << std::endl;
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

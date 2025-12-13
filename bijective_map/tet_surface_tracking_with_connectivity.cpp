#include "tet_surface_tracking_with_connectivity.hpp"
#include <CGAL/number_utils.h>
#include <chrono>
#include <cmath>
#include <cstring>
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
#include "tet_surface_tracking_internal.hpp"
#include "tet_track_operations.hpp"
#include "vtu_utils.hpp"

namespace tet_surface_tracking_with_connectivity {

// Internal helper function for performing final autorefine on the before mesh
static void perform_final_autorefine(
    query_surface_tet_with_connectivity& query_surface,
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before)
{
    std::cout << "\n=== Performing final autorefine on before mesh ===" << std::endl;
    // Convert V_before to rational
    MatrixXr V_before_rational(V_before.rows(), V_before.cols());
    for (int i = 0; i < V_before.rows(); i++) {
        for (int j = 0; j < V_before.cols(); j++) {
            V_before_rational(i, j) = wmtk::Rational(V_before(i, j));
        }
    }
    // Build sampled_points from query_surface
    // Since T_before uses global IDs directly, pt.t_id == tet_index and pt.tv_ids == vertex
    // indices
    std::set<int> unique_point_indices_set;
    for (const auto& tri : query_surface.query_triangles) {
        unique_point_indices_set.insert(tri[0]);
        unique_point_indices_set.insert(tri[1]);
        unique_point_indices_set.insert(tri[2]);
    }
    std::vector<int> unique_point_indices(
        unique_point_indices_set.begin(),
        unique_point_indices_set.end());
    std::map<int, int> global_to_local_point_map;
    for (int local_idx = 0; local_idx < static_cast<int>(unique_point_indices.size());
         local_idx++) {
        global_to_local_point_map[unique_point_indices[local_idx]] = local_idx;
    }
    // Build local_triangles_F
    Eigen::MatrixXi local_triangles_F(query_surface.query_triangles.size(), 3);
    for (size_t i = 0; i < query_surface.query_triangles.size(); i++) {
        const Eigen::Vector3i& global_tri = query_surface.query_triangles[i];
        local_triangles_F(i, 0) = global_to_local_point_map[global_tri[0]];
        local_triangles_F(i, 1) = global_to_local_point_map[global_tri[1]];
        local_triangles_F(i, 2) = global_to_local_point_map[global_tri[2]];
    }
    // Build sampled_points - T_before uses global IDs directly
    std::vector<cgal_autorefine_demo::SampledPointInputRational> sampled_points;
    sampled_points.reserve(unique_point_indices.size());
    for (int local_idx = 0; local_idx < static_cast<int>(unique_point_indices.size());
         local_idx++) {
        int global_idx = unique_point_indices[local_idx];
        const auto& pt = query_surface.points[global_idx];
        cgal_autorefine_demo::SampledPointInputRational sampled_pt;
        // pt.t_id is the global tet_id which is also the index in T_before
        sampled_pt.tet_index = static_cast<int>(pt.t_id);
        sampled_pt.barycentric = pt.bc;
        if (sampled_pt.tet_index < 0 || sampled_pt.tet_index >= T_before.rows()) {
            std::cerr << "Warning: Invalid tet_id " << pt.t_id << " for point " << global_idx
                      << std::endl;
            continue;
        }
        sampled_points.push_back(sampled_pt);
    }
    std::cout << "  Prepared " << sampled_points.size() << " sampled points and "
              << local_triangles_F.rows() << " triangles for autorefine" << std::endl;
    // Call autorefine
    auto autorefine_start = std::chrono::high_resolution_clock::now();
    cgal_autorefine_demo::AutorefineResultRational autorefine_result =
        cgal_autorefine_demo::autorefine_sampled_triangles_rational(
            V_before_rational,
            T_before,
            sampled_points,
            local_triangles_F);
    auto autorefine_end = std::chrono::high_resolution_clock::now();
    auto autorefine_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(autorefine_end - autorefine_start);
    std::cout << "  Autorefine completed: " << autorefine_result.refined_points.size()
              << " refined points, " << autorefine_result.refined_triangles.size()
              << " refined triangles" << std::endl;
    std::cout << "  Autorefine took " << autorefine_duration.count() << " ms" << std::endl;
    std::cout << "  Sampled fragment triangles: "
              << autorefine_result.sampled_fragment_triangles.size() << std::endl;
    // Update query_surface with autorefine result (without rounding and simplify)
    if (!autorefine_result.sampled_fragment_triangles.empty()) {
        std::cout << "  Updating query_surface with autorefine result..." << std::endl;
        // Map original points to surface points
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
        for (const auto& [orig_idx, surf_idx] : original_point_to_surface_point) {
            refined_point_to_surface_point[orig_idx] = surf_idx;
        }
        // Collect vertex ids used by sampled triangles
        std::set<std::size_t> refined_vertex_ids_used;
        for (const auto& tri : autorefine_result.sampled_fragment_triangles) {
            refined_vertex_ids_used.insert(tri[0]);
            refined_vertex_ids_used.insert(tri[1]);
            refined_vertex_ids_used.insert(tri[2]);
        }
        // Add new points
        for (std::size_t refined_v_id : refined_vertex_ids_used) {
            if (refined_point_to_surface_point.find(refined_v_id) !=
                refined_point_to_surface_point.end()) {
                continue;
            }
            const cgal_autorefine_demo::RationalPoint& p =
                autorefine_result.refined_points[refined_v_id];
            Vector3r point_pos;
            point_pos(0) = wmtk::Rational(p.x(), false);
            point_pos(1) = wmtk::Rational(p.y(), false);
            point_pos(2) = wmtk::Rational(p.z(), false);
            int tet_id = -1;
            if (refined_v_id < autorefine_result.vertex_tet_sets.size()) {
                const auto& tet_set = autorefine_result.vertex_tet_sets[refined_v_id];
                if (!tet_set.empty()) {
                    tet_id = *tet_set.begin();
                }
            }
            if (tet_id == -1) {
                for (std::size_t tri_idx = 0;
                     tri_idx < autorefine_result.sampled_fragment_triangles.size();
                     ++tri_idx) {
                    const auto& tri = autorefine_result.sampled_fragment_triangles[tri_idx];
                    if (tri[0] == refined_v_id || tri[1] == refined_v_id ||
                        tri[2] == refined_v_id) {
                        tet_id = autorefine_result.sampled_fragment_tet_ids(tri_idx);
                        break;
                    }
                }
            }
            if (tet_id == -1 || tet_id >= T_before.rows()) {
                std::cerr << "Warning: Could not find valid tet_id for new point " << refined_v_id
                          << std::endl;
                continue;
            }
            // T_before uses global IDs directly, so tet_id is global_tet_id
            // and T_before.row(tet_id) contains global vertex IDs
            Eigen::Vector4i tv_ids = T_before.row(tet_id);
            Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices;
            for (int i = 0; i < 4; ++i) {
                int v_id = tv_ids(i);
                if (v_id >= 0 && v_id < V_before_rational.rows()) {
                    tet_vertices.row(i) = V_before_rational.row(v_id);
                }
            }
            Vector4r barycentric_coords =
                world_to_barycentric_tet<wmtk::Rational>(point_pos, tet_vertices);
            query_point_tet_r new_point;
            new_point.t_id = tet_id; // tet_id is already global
            new_point.bc = barycentric_coords; // No rounding
            new_point.tv_ids = tv_ids; // tv_ids are already global
            std::size_t new_surface_point_idx = query_surface.points.size();
            query_surface.points.push_back(new_point);
            refined_point_to_surface_point[refined_v_id] = new_surface_point_idx;
        }
        // Clear and rebuild triangles
        query_surface.query_triangles.clear();
        query_surface.tet_ids.clear();
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
                    all_mapped = false;
                    break;
                }
            }
            if (!all_mapped) {
                continue;
            }
            query_surface.query_triangles.push_back(new_tri);
            // tet_id from autorefine is already global
            int tet_id = autorefine_result.sampled_fragment_tet_ids(i);
            query_surface.tet_ids.push_back(tet_id);
        }
        std::cout << "  Updated query_surface: " << query_surface.points.size() << " points, "
                  << query_surface.query_triangles.size() << " triangles" << std::endl;
    }
    std::cout << "=== Final autorefine completed ===" << std::endl;
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


void run_backward_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_before,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold,
    int start_operation,
    int save_interval,
    const std::filesystem::path& save_dir,
    bool do_rounding,
    bool do_simplify,
    bool only_do_arrangement_once)
{
    std::cout << "Backward tracking surface with connectivity" << std::endl;
    if (start_operation > 0) {
        std::cout << "Starting from operation " << start_operation << std::endl;
    }
    if (save_interval > 0 && !save_dir.empty()) {
        std::cout << "Saving surface every " << save_interval << " operations to " << save_dir
                  << std::endl;
    }

    // Step 1: Read or sample the query surface with connectivity
    std::string query_surface_filename = surface_file.string();
    query_surface_tet_with_connectivity query_surface;
    bool is_checkpoint_loaded = false;
    if (start_operation > 0 && !save_dir.empty()) {
        BatchOperationLogReader temp_reader(operation_logs_dir);
        int temp_total_ops = temp_reader.get_total_operations();
        if (start_operation < temp_total_ops) {
            int checkpoint_op_index = temp_total_ops - 1 - start_operation;
            std::filesystem::path checkpoint_file =
                save_dir / ("surface_op_" + std::to_string(checkpoint_op_index) + ".bin");
            if (std::filesystem::exists(checkpoint_file)) {
                std::cout << "Loading checkpoint from operation index " << checkpoint_op_index
                          << ": " << checkpoint_file << std::endl;
                query_surface = read_surface_connectivity_from_binary(checkpoint_file.string());
                is_checkpoint_loaded = true;
            }
        }
    }
    if (!is_checkpoint_loaded) {
        start_operation = 0;
        std::cout << "No checkpoint found, starting from operation 0" << std::endl;
        if (!std::filesystem::exists(query_surface_filename)) {
            std::cout << "query_surface not found, sampling and writing to file..." << std::endl;
            int N = 5;
            int axis = 2;
            double min_coord = V_after.col(axis).minCoeff();
            double max_coord = V_after.col(axis).maxCoeff();
            double plane_coord = min_coord + (max_coord - min_coord) * (N + 1) / (N + 2);
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
    }


    std::string model_name = "model";
    auto dir_str = operation_logs_dir.filename().string();
    const std::string prefix = "operation_log_";
    if (dir_str.size() > prefix.size() && dir_str.substr(0, prefix.size()) == prefix) {
        model_name = dir_str.substr(prefix.size());
    }
    if (!is_checkpoint_loaded) {
        write_surface_to_vtu(
            query_surface,
            V_after,
            model_name + "_query_surface_tet_with_connectivity_after.vtu");
    }
    if (!is_checkpoint_loaded) {
        // DEBUG: sanitity check for the input query_surface
        for (int i = 0; i < query_surface.query_triangles.size(); i++) {
            const Eigen::Vector3i& tri = query_surface.query_triangles[i];
            int tri_tet_id = query_surface.tet_ids[i];

            const Eigen::Vector4i& relevant_vids = T_after.row(tri_tet_id);

            // check points
            for (int j = 0; j < 3; j++) {
                const auto& pt = query_surface.points[tri[j]];
                if (pt.t_id != tri_tet_id) {
                    // std::cout << "i: " << i << ", tri_tet_id: " << tri_tet_id
                    //           << ", pt: [t_id=" << pt.t_id << ", bc=(" << pt.bc[0].to_double()
                    //           << ", " << pt.bc[1].to_double() << ", " << pt.bc[2].to_double()
                    //           << ", " << pt.bc[3].to_double() << ")]" << std::endl;

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

    // check santity of the input query_surface
    {
        bool is_manifold = check_surface_manifold_property(query_surface.query_triangles);
        if (is_manifold) {
            std::cout << "Input query surface is manifold" << std::endl;
        } else {
            std::cout << "Input query surface is not manifold" << std::endl;
            throw std::runtime_error("Error: input query_surface is not manifold");
        }
    }

    // Step 2: Do the backward tracking
    std::cout << "Doing backward tracking..." << std::endl;
    BatchOperationLogReader reader(operation_logs_dir);
    int total_ops = reader.get_total_operations();
    if (total_ops == 0) {
        std::cerr << "No operation logs found in " << operation_logs_dir << std::endl;
        return;
    }
    std::cout << "Found " << total_ops << " operations in "
              << (reader.is_batch_format() ? "batch" : "legacy") << " format" << std::endl;
    if (start_operation >= total_ops) {
        std::cerr << "start_operation " << start_operation << " is >= total_ops " << total_ops
                  << std::endl;
        return;
    }
    bool do_forward = false;
    int ops_to_process = total_ops - start_operation;
    if (!save_dir.empty()) {
        std::filesystem::create_directories(save_dir);
    }
    auto tracking_block_start = std::chrono::high_resolution_clock::now();
    for (int i = start_operation; i < total_ops; ++i) {
        int operation_index = i;
        if (!do_forward) {
            operation_index = total_ops - 1 - i;
        }
        nlohmann::json operation_log = reader.get_operation(operation_index);
        if (operation_log.empty()) {
            std::cerr << "Failed to read operation " << operation_index << std::endl;
            throw std::runtime_error("Error: failed to read operation");
        }
        int current_op = i - start_operation + 1;
        std::cout << "\n=== Processing operation " << current_op << "/" << ops_to_process
                  << " (index: " << operation_index << ") ===" << std::endl;
        track_one_operation(
            operation_log,
            query_surface,
            do_forward,
            static_cast<int>(operation_index),
            do_rounding,
            do_simplify,
            only_do_arrangement_once);
        if (save_interval > 0 && !save_dir.empty() &&
            (current_op % save_interval == 0 || current_op == ops_to_process)) {
            std::filesystem::path save_file =
                save_dir / ("surface_op_" + std::to_string(operation_index) + ".bin");
            write_surface_connectivity_to_binary(query_surface, save_file.string());
        }
    }
    auto tracking_block_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tracking_block_duration =
        tracking_block_end - tracking_block_start;
    std::cout << "Total tracking block took " << tracking_block_duration.count() << " seconds"
              << std::endl;


    // TODO: a experimental version of only do arrangement once
    if (only_do_arrangement_once) {
        perform_final_autorefine(query_surface, V_before, T_before);
    }

    std::cout << "\n=== All operations completed ===" << std::endl;
    std::cout << "Final surface state: " << query_surface.points.size() << " points, "
              << query_surface.query_triangles.size() << " triangles" << std::endl;

    // step3 write the surface to file
    write_surface_connectivity_to_file(
        query_surface,
        model_name + "_query_surface_tet_with_connectivity_before.json");
    write_surface_to_vtu(
        query_surface,
        V_before,
        model_name + "_query_surface_tet_with_connectivity_before.vtu");

    // results manifold check
    {
        bool is_manifold = check_surface_manifold_property(query_surface.query_triangles);
        if (is_manifold) {
            std::cout << "Output query surface is manifold" << std::endl;
        } else {
            std::cout << "Output query surface is not manifold" << std::endl;
            throw std::runtime_error("Error: output query_surface is not manifold");
        }
    }

    // check self intersection
    // TODO: make it a intrinsic one
    {
        // MatrixXr V_before_rational(V_before.rows(), V_before.cols());
        // for (int i = 0; i < V_before.rows(); i++) {
        //     for (int j = 0; j < V_before.cols(); j++) {
        //         V_before_rational(i, j) = wmtk::Rational(V_before(i, j));
        //     }
        // }
        // auto [surface_V, surface_F_matrix] =
        //     surface_to_world_positions_rational(query_surface,
        //     V_before_rational);self_intersection
        // bool has_ =
        //     check_surface_self_intersection(surface_V, query_surface.query_triangles);
        // if (has_self_intersection) {
        //     std::cout << "Output query surface has self-intersection" << std::endl;
        //     throw std::runtime_error("Error: output query_surface has self-intersection");
        // } else {
        //     std::cout << "Output query surface has no self-intersection" << std::endl;
        // }
        bool has_self_intersection =
            check_surface_self_intersection_intrinsic(query_surface, T_before);
        if (has_self_intersection) {
            std::cout << "Output query surface has self-intersection" << std::endl;
            throw std::runtime_error("Error: output query_surface has self-intersection");
        } else {
            std::cout << "Output query surface has no self-intersection" << std::endl;
        }
    }
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

void write_surface_connectivity_to_binary(
    const query_surface_tet_with_connectivity& surface,
    const std::string& filename)
{
    std::cout << "Writing surface with connectivity to binary file: " << filename << std::endl;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    const char* magic = "SURF";
    file.write(magic, 4);
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    size_t num_points = surface.points.size();
    size_t num_triangles = surface.query_triangles.size();
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
    file.write(reinterpret_cast<const char*>(&num_triangles), sizeof(num_triangles));
    for (size_t i = 0; i < num_points; ++i) {
        const auto& pt = surface.points[i];
        file.write(reinterpret_cast<const char*>(&pt.t_id), sizeof(pt.t_id));
        for (int j = 0; j < 4; ++j) {
            std::string bc_str = pt.bc[j].serialize();
            uint32_t bc_len = static_cast<uint32_t>(bc_str.size());
            file.write(reinterpret_cast<const char*>(&bc_len), sizeof(bc_len));
            file.write(bc_str.c_str(), bc_len);
        }
        for (int j = 0; j < 4; ++j) {
            int32_t vid = static_cast<int32_t>(pt.tv_ids[j]);
            file.write(reinterpret_cast<const char*>(&vid), sizeof(vid));
        }
    }
    for (size_t i = 0; i < num_triangles; ++i) {
        const auto& tri = surface.query_triangles[i];
        for (int j = 0; j < 3; ++j) {
            int32_t idx = static_cast<int32_t>(tri[j]);
            file.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        }
        int32_t tet_id =
            (i < surface.tet_ids.size()) ? static_cast<int32_t>(surface.tet_ids[i]) : -1;
        file.write(reinterpret_cast<const char*>(&tet_id), sizeof(tet_id));
    }
    file.close();
    std::cout << "Successfully wrote " << num_points << " points and " << num_triangles
              << " triangles to binary file " << filename << std::endl;
}

query_surface_tet_with_connectivity read_surface_connectivity_from_binary(
    const std::string& filename)
{
    std::cout << "Reading surface with connectivity from binary file: " << filename << std::endl;
    query_surface_tet_with_connectivity surface;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return surface;
    }
    char magic[5] = {0};
    file.read(magic, 4);
    if (std::strcmp(magic, "SURF") != 0) {
        std::cerr << "Invalid magic number in binary file: " << filename << std::endl;
        file.close();
        return surface;
    }
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "Unsupported version " << version << " in binary file: " << filename
                  << std::endl;
        file.close();
        return surface;
    }
    size_t num_points, num_triangles;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    file.read(reinterpret_cast<char*>(&num_triangles), sizeof(num_triangles));
    surface.points.resize(num_points);
    surface.query_triangles.resize(num_triangles);
    surface.tet_ids.resize(num_triangles);
    for (size_t i = 0; i < num_points; ++i) {
        auto& pt = surface.points[i];
        file.read(reinterpret_cast<char*>(&pt.t_id), sizeof(pt.t_id));
        for (int j = 0; j < 4; ++j) {
            uint32_t bc_len;
            file.read(reinterpret_cast<char*>(&bc_len), sizeof(bc_len));
            std::string bc_str(bc_len, '\0');
            file.read(&bc_str[0], bc_len);
            pt.bc[j] = wmtk::Rational::deserialize(bc_str);
        }
        for (int j = 0; j < 4; ++j) {
            int32_t vid;
            file.read(reinterpret_cast<char*>(&vid), sizeof(vid));
            pt.tv_ids[j] = vid;
        }
    }
    for (size_t i = 0; i < num_triangles; ++i) {
        auto& tri = surface.query_triangles[i];
        for (int j = 0; j < 3; ++j) {
            int32_t idx;
            file.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            tri[j] = idx;
        }
        int32_t tet_id;
        file.read(reinterpret_cast<char*>(&tet_id), sizeof(tet_id));
        surface.tet_ids[i] = tet_id;
    }
    file.close();
    std::cout << "Successfully read " << num_points << " points and " << num_triangles
              << " triangles from binary file " << filename << std::endl;
    return surface;
}

} // namespace tet_surface_tracking_with_connectivity

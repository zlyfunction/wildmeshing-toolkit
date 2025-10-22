#pragma once

#include <Eigen/Core>
#include <filesystem>
#include "tet_track_operations.hpp"

namespace tet_surface_tracking {

// Convert query surface to world positions
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> query_surface_to_world_positions(
    const query_surface_tet& query_surface,
    const Eigen::MatrixXd& V);

// Print triangle area statistics
void print_triangle_area_statistics(
    const Eigen::MatrixXd& surface_V,
    const Eigen::MatrixXi& surface_F);

// Check if the surface is manifold
void check_manifold_property(
    const Eigen::MatrixXd& surface_V,
    const Eigen::MatrixXi& surface_F);

// Run back-tracking surface application
void run_back_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold = false);

// Surface I/O functions
void write_query_surface_tet_to_file(
    const query_surface_tet& surface,
    const std::string& filename);

query_surface_tet read_query_surface_tet_from_file(const std::string& filename);

// Surface tracking core functions
void handle_consolidate_tet_surface(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet& surface,
    bool forward);

void handle_local_mapping_tet_surface(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_surface_tet& surface);

void track_surface_one_operation_tet(
    const nlohmann::json& operation_log,
    query_surface_tet& surface,
    bool do_forward,
    bool use_rational,
    int operation_id);

void track_surface_tet(
    const std::filesystem::path& dirPath,
    query_surface_tet& surface,
    bool do_forward,
    bool use_rational);

} // namespace tet_surface_tracking

#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>
#include "tet_track_operations.hpp"

namespace tet_point_tracking {

// Write query points to VTU file and return computed coordinates
Eigen::MatrixXd write_points_to_file(
    const std::vector<query_point_tet>& query_points,
    const Eigen::MatrixXd& V,
    const std::string& filename);

// Run back-tracking point application
void run_back_tracking(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::string& points_after_remesh_filename = "points_after_remesh.vtu",
    const std::string& points_after_tracking_filename = "points_after_back_tracking.vtu");

// Point tracking core functions
void handle_consolidate_tet(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    std::vector<query_point_tet>& query_points,
    bool forward);

void handle_local_mapping_tet(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    std::vector<query_point_tet>& query_points);

void track_point_one_operation_tet(
    const nlohmann::json& operation_log,
    std::vector<query_point_tet>& query_points,
    bool do_forward,
    bool use_rational,
    int operation_id);

void track_point_tet(
    const std::filesystem::path& dirPath,
    std::vector<query_point_tet>& query_points,
    bool do_forward,
    bool use_rational);

} // namespace tet_point_tracking

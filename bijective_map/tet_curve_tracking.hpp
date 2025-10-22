#pragma once

#include <Eigen/Core>
#include <filesystem>
#include "tet_track_operations.hpp"

namespace tet_curve_tracking {

// Write curve points to CSV files
void write_curve_points_to_file(
    const Eigen::MatrixXd& V,
    const query_curve_tet& curve,
    const std::string& filename1,
    const std::string& filename2);

// Run back-tracking curve application
void run_back_tracking_curve(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir);

// Curve tracking core functions
void handle_consolidate_tet_curve(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_curve_tet& curve,
    bool forward);

void handle_one_segment_tet(
    query_curve_tet& curve,
    int id,
    std::vector<query_point_tet>& current_qps,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T,
    const std::vector<int64_t>& id_map,
    const std::vector<int64_t>& v_id_map,
    Eigen::MatrixXi& TT,
    Eigen::MatrixXi& TTi,
    double eps);

void handle_local_mapping_tet_curve(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_curve_tet& curve);

void track_curve_one_operation_tet(
    const nlohmann::json& operation_log,
    query_curve_tet& curve,
    bool do_forward,
    bool use_rational,
    int operation_id);

void track_curve_tet(
    const std::filesystem::path& dirPath,
    query_curve_tet& curve,
    bool do_forward,
    bool use_rational);

} // namespace tet_curve_tracking

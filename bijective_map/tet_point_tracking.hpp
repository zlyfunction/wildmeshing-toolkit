#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>
#include "track_operations_tet.hpp"

namespace tet_point_tracking {

// Write query points to CSV file
void write_points_to_file(
    const std::vector<query_point_tet>& query_points,
    const Eigen::MatrixXd& V,
    const std::string& filename);

// Run back-tracking point application
void run_back_tracking(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir);

} // namespace tet_point_tracking

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

} // namespace tet_curve_tracking

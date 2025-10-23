#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <string>
#include <vector>
#include "tet_track_operations.hpp"

namespace tet_point_tracking {

// Write query points to VTU file and return computed coordinates
template <typename CoordType = double>
Eigen::MatrixXd write_points_to_file(
    const std::vector<query_point_tet_t<CoordType>>& query_points,
    const Eigen::MatrixXd& V,
    const std::string& filename);

extern template Eigen::MatrixXd write_points_to_file<double>(
    const std::vector<query_point_tet_t<double>>&,
    const Eigen::MatrixXd&,
    const std::string&);
extern template Eigen::MatrixXd write_points_to_file<wmtk::Rational>(
    const std::vector<query_point_tet_t<wmtk::Rational>>&,
    const Eigen::MatrixXd&,
    const std::string&);

// Run back-tracking point application
void run_back_tracking(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::string& points_after_remesh_filename = "points_after_remesh.vtu",
    const std::string& points_after_tracking_filename = "points_after_back_tracking.vtu");

// Run back-tracking point application with rational arithmetic
void run_back_tracking_rational(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::string& points_after_remesh_filename = "points_after_remesh_rational.vtu",
    const std::string& points_after_tracking_filename = "points_after_back_tracking_rational.vtu");

} // namespace tet_point_tracking

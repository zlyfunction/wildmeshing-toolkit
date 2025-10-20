#pragma once

#include <Eigen/Core>
#include <filesystem>
#include "track_operations_tet.hpp"

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

} // namespace tet_surface_tracking

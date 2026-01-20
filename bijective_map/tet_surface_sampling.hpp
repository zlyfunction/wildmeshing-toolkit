#pragma once

#include <Eigen/Core>
#include "tet_track_operations.hpp"

namespace tet_surface_sampling {

// Sample a large triangle using CGAL arrangement
query_surface_tet sample_query_surface_large_triangle(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out);

// Sample a query surface with connectivity
query_surface_tet_with_connectivity sample_query_surface_tet_with_connectivity(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out);

// Build a query surface with connectivity from a triangle mesh embedded in a tet mesh.
query_surface_tet_with_connectivity query_surface_tet_with_connectivity_from_triangle_mesh(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    double tolerance = 1e-8,
    bool verbose = false);

// Refine a triangle mesh against tet mesh boundaries and return the arranged triangle mesh.
bool arrangement_triangle_mesh_in_tet_mesh(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    Eigen::MatrixXd& V_arranged,
    Eigen::MatrixXi& F_arranged,
    double tolerance = 1e-8,
    bool verbose = false);

// Build query surface by computing per-vertex tet + barycentric coordinates without arrangement.
query_surface_tet_with_connectivity query_surface_tet_with_connectivity_no_arrangement(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXd& V_surface,
    const Eigen::MatrixXi& F_surface,
    bool verbose = false);

// Slice the tet mesh with an axis-aligned plane (axis = 0/1/2 for x/y/z, coordinate = constant)
query_surface_tet_with_connectivity slice_tet_mesh_with_axis_plane(
    const Eigen::MatrixXi& T,
    const Eigen::MatrixXd& V,
    int axis,
    double constant);

// Sample a sub-surface by traversing connected tetrahedrons
query_surface_tet sample_query_surface_sub_surface(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out);

} // namespace tet_surface_sampling

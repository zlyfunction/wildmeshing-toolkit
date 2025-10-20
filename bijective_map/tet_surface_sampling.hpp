#pragma once

#include <Eigen/Core>
#include "track_operations_tet.hpp"

namespace tet_surface_sampling {

// Sample a large triangle using CGAL arrangement
query_surface_tet sample_query_surface_large_triangle(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out);

// Sample a sub-surface by traversing connected tetrahedrons
query_surface_tet sample_query_surface_sub_surface(
    const Eigen::MatrixXi& T_out,
    const Eigen::MatrixXd& V_out);

} // namespace tet_surface_sampling

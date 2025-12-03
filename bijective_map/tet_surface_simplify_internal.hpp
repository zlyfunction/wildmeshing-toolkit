#pragma once

#include <Eigen/Core>
#include <wmtk/utils/Rational.hpp>
#include "tet_track_operations.hpp"

namespace tet_surface_tracking_with_connectivity {

using MatrixXr = Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Simplify refined triangles by grouping them by tet_id and removing interior points
 * For each tet_id's subsurface, if it has interior points, extract boundary loop
 * and retriangulate using ear clipping
 * This function processes triangles starting from start_tri_idx
 *
 * @param surface The query surface to simplify
 * @param start_tri_idx Starting triangle index for refined triangles
 * @param V_before Vertex positions before operation (rational)
 * @param T_before Tetrahedral connectivity before operation
 * @param id_map_before Mapping from global tet ID to local index in T_before
 * @param v_id_map_before Mapping from global vertex ID to local index in V_before
 * @param operation_id Operation ID for file naming
 */
void simplify_refined_triangles_by_tet(
    query_surface_tet_with_connectivity& surface,
    size_t start_tri_idx,
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    int operation_id);

} // namespace tet_surface_tracking_with_connectivity

#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>
#include <wmtk/utils/Rational.hpp>
#include "tet_track_operations.hpp"

namespace tet_surface_tracking_with_connectivity {

// Type aliases for rational matrices
using MatrixXr = Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, Eigen::Dynamic>;
using Vector3r = Eigen::Matrix<wmtk::Rational, 3, 1>;
using Vector4r = Eigen::Matrix<wmtk::Rational, 4, 1>;

/**
 * @brief Convert query surface with connectivity to world positions using rational coordinates
 *
 * @param query_surface The query surface with connectivity structure
 * @param V Vertex positions matrix (n x 3) with rational coordinates
 * @return std::pair<Eigen::MatrixXr, Eigen::MatrixXi> Vertex positions and triangle indices
 */
std::pair<MatrixXr, Eigen::MatrixXi> surface_to_world_positions_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V);

/**
 * @brief Check if the surface is manifold
 *
 * @param surface_F Surface triangle indices
 * @return bool True if the surface is manifold (both edge and vertex manifold)
 */
bool check_surface_manifold_property(const Eigen::MatrixXi& surface_F);

/**
 * @brief Check if the surface has self-intersection
 *
 * @param surface_V Surface vertex positions with rational coordinates
 * @param surface_F Surface triangle indices
 */
void check_surface_self_intersection(const MatrixXr& surface_V, const Eigen::MatrixXi& surface_F);


/**
 * @brief Handle consolidate operation for surface with connectivity
 *
 * @param tet_ids_maps Mapping of tetrahedron IDs
 * @param vertex_ids_maps Mapping of vertex IDs
 * @param surface The query surface to update
 * @param forward Whether to apply forward or backward mapping
 */
void handle_consolidate_operation(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet_with_connectivity& surface,
    bool forward);

/**
 * @brief Arrange surface triangles using autorefine
 *
 * @param V_before Vertex positions before operation (rational)
 * @param T_before Tetrahedral connectivity before operation
 * @param id_map_before Tetrahedron ID mapping before operation
 * @param v_id_map_before Vertex ID mapping before operation
 * @param id_map_after Tetrahedron ID mapping after operation
 * @param surface The query surface to update
 * @param operation_id Operation ID for file naming
 * @param do_rounding Whether to round barycentric coordinates
 * @param verbose Whether to print verbose output
 * @param save_debug_meshes Whether to save debug mesh files
 */
void surface_triangle_arrangement(
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const std::vector<int64_t>& id_map_after,
    query_surface_tet_with_connectivity& surface,
    int operation_id,
    bool do_rounding = true,
    bool verbose = false,
    bool save_debug_meshes = false);

/**
 * @brief Handle local mapping operation for surface with connectivity
 *
 * @param V_before Vertex positions before operation (rational)
 * @param T_before Tetrahedral connectivity before operation
 * @param id_map_before Tetrahedron ID mapping before operation
 * @param v_id_map_before Vertex ID mapping before operation
 * @param V_after Vertex positions after operation (rational)
 * @param T_after Tetrahedral connectivity after operation
 * @param id_map_after Tetrahedron ID mapping after operation
 * @param v_id_map_after Vertex ID mapping after operation
 * @param surface The query surface to update
 * @param operation_id Operation ID for file naming
 */
void handle_local_mapping_operation(
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const MatrixXr& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_surface_tet_with_connectivity& surface,
    int operation_id);

/**
 * @brief Track surface through one operation
 *
 * @param operation_log JSON operation log
 * @param query_surface The query surface to track
 * @param do_forward Whether to track forward or backward
 * @param operation_id Operation ID for debugging
 */
void track_one_operation(
    const nlohmann::json& operation_log,
    query_surface_tet_with_connectivity& surface,
    bool do_forward,
    int operation_id);

/**
 * @brief Track surface through all operations in a directory
 *
 * @param dirPath Directory containing operation logs
 * @param query_surface The query surface to track
 * @param do_forward Whether to track forward or backward
 */
void track_all_operations(
    const std::filesystem::path& dirPath,
    query_surface_tet_with_connectivity& surface,
    bool do_forward);

/**
 * @brief Helper function to get all possible representations of a point in local patch
 *
 * @param local_t_id The local tetrahedron ID
 * @param local_bc The barycentric coordinates within the tetrahedron (rational)
 * @param T_local The local tetrahedron connectivity matrix
 * @return std::pair<std::vector<int>, std::vector<Eigen::Vector4r>> All possible tetrahedron IDs
 *         and their corresponding barycentric coordinates
 */
std::pair<std::vector<int>, std::vector<Vector4r>> get_point_representations(
    const int local_t_id,
    const Vector4r& local_bc,
    const Eigen::MatrixXi& T_local);

} // namespace tet_surface_tracking_with_connectivity

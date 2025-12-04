#pragma once

#include <Eigen/Core>
#include <filesystem>
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
 * @brief Write query surface with connectivity to VTU file (rational version)
 *
 * @param query_surface The query surface with connectivity structure
 * @param V Vertex positions matrix (n x 3) with rational coordinates
 * @param filename Output VTU filename
 */
void write_surface_to_vtu_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V,
    const std::string& filename);

/**
 * @brief Write query surface with connectivity to VTU file (double version)
 *
 * @param query_surface The query surface with connectivity structure
 * @param V Vertex positions matrix (n x 3) with double coordinates
 * @param filename Output VTU filename
 */
void write_surface_to_vtu(
    const query_surface_tet_with_connectivity& query_surface,
    const Eigen::MatrixXd& V,
    const std::string& filename);
/**
 * @brief Run back-tracking surface application
 *
 * @param T_after Tetrahedral connectivity after operations
 * @param V_after Vertex positions after operations
 * @param T_before Tetrahedral connectivity before operations
 * @param V_before Vertex positions before operations
 * @param operation_logs_dir Directory containing operation logs
 * @param surface_file Path to surface file
 * @param check_manifold Whether to check manifold property (default: false)
 * @param start_operation Start from operation N (default: 0)
 * @param save_interval Save surface every N operations (default: 1, save every operation)
 * @param save_dir Directory to save intermediate surfaces (empty = don't save)
 * @param do_rounding Whether to round barycentric coordinates during tracking (default: false)
 * @param do_simplify Whether to simplify refined triangles by removing interior points (default: false)
 * @param only_do_arrangement_once Whether to perform final autorefine on before mesh (default: true)
 */
void run_backward_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_before,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold = false,
    int start_operation = 0,
    int save_interval = 1,
    const std::filesystem::path& save_dir = std::filesystem::path(),
    bool do_rounding = false,
    bool do_simplify = false,
    bool only_do_arrangement_once = false);

/**
 * @brief Write query surface with connectivity to file
 *
 * @param surface The query surface with connectivity
 * @param filename Output filename
 */
void write_surface_connectivity_to_file(
    const query_surface_tet_with_connectivity& surface,
    const std::string& filename);

/**
 * @brief Read query surface with connectivity from file
 *
 * @param filename Input filename
 * @return query_surface_tet_with_connectivity The loaded surface
 */
query_surface_tet_with_connectivity read_surface_connectivity_from_file(
    const std::string& filename);

/**
 * @brief Write query surface with connectivity to binary file (preserves full precision)
 *
 * @param surface The query surface with connectivity
 * @param filename Output binary filename
 */
void write_surface_connectivity_to_binary(
    const query_surface_tet_with_connectivity& surface,
    const std::string& filename);

/**
 * @brief Read query surface with connectivity from binary file
 *
 * @param filename Input binary filename
 * @return query_surface_tet_with_connectivity The loaded surface
 */
query_surface_tet_with_connectivity read_surface_connectivity_from_binary(
    const std::string& filename);

} // namespace tet_surface_tracking_with_connectivity

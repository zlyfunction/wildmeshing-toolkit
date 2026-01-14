#pragma once
#include <wmtk/Mesh.hpp>
#include <wmtk/TetMesh.hpp>
#include "TetRemeshingOptions.hpp"
namespace wmtk::components::tet_remeshing {
/**
 * @brief Perform tet remeshing (split and collapse) on a tetrahedral mesh.
 *
 * This function generates new attributes that are not removed automatically.
 *
 * @param mesh The root mesh (must be a TetMesh).
 * @param options All options required for performing the tet remeshing.
 */
void tet_remeshing(Mesh& mesh, const TetRemeshingOptions& options);
/**
 * @brief Perform tet remeshing on a tetrahedral mesh.
 *
 * This function generates new attributes that are not removed automatically.
 *
 * This function wraps the behavior of `tet_remeshing` with options. For details on the
 * default values of the options, look at TetRemeshingOptions.
 *
 */
void tet_remeshing(
    Mesh& mesh,
    const attribute::MeshAttributeHandle& position_handle,
    const double length_rel,
    std::optional<bool> lock_boundary = {},
    std::optional<double> envelope_size = {},
    bool check_inversion = false,
    bool enable_split = true,
    bool enable_collapse = true,
    bool enable_swap = true,
    int iterations = 10,
    const std::vector<attribute::MeshAttributeHandle>& pass_through = {});
} // namespace wmtk::components::tet_remeshing

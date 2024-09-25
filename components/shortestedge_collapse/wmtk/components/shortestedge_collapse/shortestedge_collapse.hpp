#pragma once
#include <wmtk/Mesh.hpp>
#include <wmtk/TriMesh.hpp>

namespace wmtk::components {

std::shared_ptr<Mesh> shortestedge_collapse(
    TriMesh& mesh,
    const attribute::MeshAttributeHandle& position_handle,
    std::optional<attribute::MeshAttributeHandle>& inversion_position_handle,
    bool update_other_position,
    const double length_rel,
    bool lock_boundary,
    double envelope_size,
    const std::vector<attribute::MeshAttributeHandle>& pass_through);

} // namespace wmtk::components

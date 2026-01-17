#pragma once

#include "tet_surface_tracking_internal.hpp"

#include <cstddef>

namespace tet_surface_tracking_with_connectivity {

struct IntersectionComponentResult
{
    std::size_t component_count = 0;
    std::size_t segment_count = 0;
    std::size_t point_count = 0;
};

IntersectionComponentResult compute_surface_intersection_components(
    const query_surface_tet_with_connectivity& surface_a,
    const query_surface_tet_with_connectivity& surface_b,
    const MatrixXr& tet_vertices,
    bool verbose = false);

} // namespace tet_surface_tracking_with_connectivity

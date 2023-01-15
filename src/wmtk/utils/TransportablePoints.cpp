
#include <wmtk/utils/TransportablePoints.hpp>


using namespace wmtk;


TransportablePointsBase::~TransportablePointsBase() = default;

void TransportablePointsBase::before_hook(const TriMesh& m, const std::set<size_t>& input_tris)
{
    std::set<size_t>& active_pts = active_points.local();
    // go through set of input tris and try to move every point in an output tri
    for (size_t point_index = 0; point_index < triangle_indices.size(); ++point_index) {
        if (input_tris.find(index) != input_tris.end()) {
            active_pts.emplace(point_index);
            update_barycetnric(m, point_index);
        }
    }
}

void TransportablePointsBase::after_hook(const TriMesh& m, const std::set<size_t>& output_tris)
{
    // go through set of input tris and try to move every point in an output tri
    for (const size_t point_index : active_points.local()) {
        // this point needs to be moved forward in this operation
        // try to see if it's in a triangle
        bool found = false;
        for (const size_t output_tri : output_tris) {
            if (point_in_triangle(m, triangle_index, point_index)) {
                triangle_indices[point_index] = triangle_index;
                barycentric_coordinates[point_index] =
                    get_barycentric(m, triangle_index, point_index);
                found = true;
            }
        }

        if (!found) {
            spdlog::warn("Point not found in a triangle, backup mechanism required");
        }
    }
}


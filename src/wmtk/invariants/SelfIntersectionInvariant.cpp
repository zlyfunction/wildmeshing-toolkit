#include "SelfIntersectionInvariant.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <Eigen/Core>
#include <wmtk/TetMesh.hpp>
#include <array>
#include <unordered_map>

namespace PMP = CGAL::Polygon_mesh_processing;

namespace wmtk {

template <typename T>
SelfIntersectionInvariant<T>::SelfIntersectionInvariant(
    const Mesh& m,
    const TypedAttributeHandle<T>& coordinate)
    : Invariant(m, true, false, true)
    , m_coordinate_handle(coordinate)
{}

template <typename T>
bool SelfIntersectionInvariant<T>::after(
    const std::vector<Tuple>&,
    const std::vector<Tuple>& top_dimension_tuples_after) const
{
    if (mesh().top_simplex_type() != PrimitiveType::Tetrahedron) {
        return true;
    }

    const TetMesh& tetmesh = static_cast<const TetMesh&>(mesh());
    const auto accessor = tetmesh.create_const_accessor(m_coordinate_handle);
    if (accessor.dimension() != 3) {
        return true;
    }

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point = Kernel::Point_3;
    using Triangle = std::array<std::size_t, 3>;

    std::vector<Point> points;
    std::vector<Triangle> triangles;
    points.reserve(top_dimension_tuples_after.size() * 4);
    triangles.reserve(top_dimension_tuples_after.size() * 4);

    std::unordered_map<int64_t, std::size_t> vertex_map;
    const auto get_vertex_index = [&](const Tuple& vtuple) {
        const int64_t gid = tetmesh.id_vertex(vtuple);
        const auto it = vertex_map.find(gid);
        if (it != vertex_map.end()) {
            return it->second;
        }
        const Eigen::Vector3<T> p = accessor.const_vector_attribute(vtuple);
        points.emplace_back(
            static_cast<double>(p.x()),
            static_cast<double>(p.y()),
            static_cast<double>(p.z()));
        const std::size_t idx = points.size() - 1;
        vertex_map.emplace(gid, idx);
        return idx;
    };

    for (const Tuple& tet : top_dimension_tuples_after) {
        const auto verts = tetmesh.orient_vertices(tet);
        if (verts.size() != 4) {
            continue;
        }

        std::array<std::size_t, 4> v_ids;
        for (int i = 0; i < 4; ++i) {
            v_ids[static_cast<std::size_t>(i)] = get_vertex_index(verts[i]);
        }

        triangles.push_back(Triangle{{v_ids[0], v_ids[1], v_ids[2]}});
        triangles.push_back(Triangle{{v_ids[0], v_ids[1], v_ids[3]}});
        triangles.push_back(Triangle{{v_ids[0], v_ids[2], v_ids[3]}});
        triangles.push_back(Triangle{{v_ids[1], v_ids[2], v_ids[3]}});
    }

    if (triangles.empty()) {
        return true;
    }

    const bool has_intersection = PMP::does_triangle_soup_self_intersect(points, triangles);
    return !has_intersection;
}

template class SelfIntersectionInvariant<double>;

} // namespace wmtk

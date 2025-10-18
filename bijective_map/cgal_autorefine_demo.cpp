#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/number_utils.h>
#include "vtu_utils.hpp"

#include <Eigen/Core>
#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;

struct TriangleTrackingVisitor : PMP::Autorefinement::Default_visitor
{
    TriangleTrackingVisitor() = default;
    explicit TriangleTrackingVisitor(std::vector<std::size_t>& mapping)
        : m_mapping(&mapping)
    {}

    void number_of_output_triangles(std::size_t nbt)
    {
        if (m_mapping == nullptr) {
            return;
        }
        m_mapping->assign(nbt, static_cast<std::size_t>(-1));
    }

    void verbatim_triangle_copy(std::size_t tgt_id, std::size_t src_id)
    {
        store_mapping(tgt_id, src_id);
    }

    void new_subtriangle(std::size_t tgt_id, std::size_t src_id)
    {
        store_mapping(tgt_id, src_id);
    }

private:
    void store_mapping(std::size_t tgt_id, std::size_t src_id)
    {
        if (m_mapping == nullptr) {
            return;
        }
        if (tgt_id >= m_mapping->size()) {
            m_mapping->resize(tgt_id + 1, static_cast<std::size_t>(-1));
        }
        (*m_mapping)[tgt_id] = src_id;
    }

    std::vector<std::size_t>* m_mapping = nullptr;
};

int main()
{
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point = Kernel::Point_3;
    using Triangle = std::array<std::size_t, 3>;

    // Two triangles that intersect along their interiors.
    std::vector<Point> points = {
        Point(-1.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(0.3, 1.0, 0.2),
        Point(0.3, -1.0, -0.2),
        Point(0.0, 0.3, -0.5),
        Point(0.0, -0.3, 0.5),
    };

    std::vector<Triangle> triangles = {
        Triangle{0, 2, 3},
        Triangle{1, 4, 5},
    };

    const auto to_vertex_matrix = [](const std::vector<Point>& pts) {
        Eigen::MatrixXd V(pts.size(), 3);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(pts.size()); ++i) {
            V(i, 0) = CGAL::to_double(pts[i].x());
            V(i, 1) = CGAL::to_double(pts[i].y());
            V(i, 2) = CGAL::to_double(pts[i].z());
        }
        return V;
    };

    const auto to_face_matrix = [](const std::vector<Triangle>& tris) {
        Eigen::MatrixXi F(tris.size(), 3);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(tris.size()); ++i) {
            F(i, 0) = static_cast<int>(tris[i][0]);
            F(i, 1) = static_cast<int>(tris[i][1]);
            F(i, 2) = static_cast<int>(tris[i][2]);
        }
        return F;
    };

    const std::vector<Point> original_points = points;
    const std::vector<Triangle> original_triangles = triangles;

    const bool had_initial_intersections =
        PMP::does_triangle_soup_self_intersect(points, triangles);

    std::vector<std::vector<std::size_t>> working_triangles;
    working_triangles.reserve(triangles.size());
    for (const Triangle& tri : triangles) {
        working_triangles.push_back({tri[0], tri[1], tri[2]});
    }

    std::vector<std::size_t> triangle_source_ids;
    TriangleTrackingVisitor visitor(triangle_source_ids);
    PMP::autorefine_triangle_soup(
        points,
        working_triangles,
        CGAL::parameters::visitor(visitor).apply_iterative_snap_rounding(true));

    triangles.clear();
    triangles.reserve(working_triangles.size());
    std::vector<std::size_t> filtered_source_ids;
    filtered_source_ids.reserve(working_triangles.size());
    for (std::size_t i = 0; i < working_triangles.size(); ++i) {
        const auto& tri = working_triangles[i];
        if (tri.size() != 3) {
            std::cerr << "Warning: encountered triangle with " << tri.size()
                      << " vertices after autorefinement; skipping.\n";
            continue;
        }
        triangles.push_back(Triangle{tri[0], tri[1], tri[2]});
        if (i < triangle_source_ids.size()) {
            filtered_source_ids.push_back(triangle_source_ids[i]);
        }
    }
    triangle_source_ids.swap(filtered_source_ids);

    const bool intersection_free = !PMP::does_triangle_soup_self_intersect(points, triangles);

    std::cout << "Initial soup had intersections: " << (had_initial_intersections ? "yes" : "no")
              << '\n';
    std::cout << "After autorefinement: "
              << (intersection_free ? "no remaining intersections." : "still intersects.") << '\n';
    std::cout << "Output point count: " << points.size() << '\n';
    std::cout << "Output triangle count: " << triangles.size() << '\n';

    if (!triangles.empty()) {
        std::cout << "First triangle indices: " << triangles.front()[0] << ", "
                  << triangles.front()[1] << ", " << triangles.front()[2] << '\n';
    }

    if (triangle_source_ids.size() != triangles.size()) {
        std::cerr << "Warning: visitor returned " << triangle_source_ids.size()
                  << " triangle mappings for " << triangles.size() << " output triangles.\n";
    }

    for (std::size_t i = 0; i < triangle_source_ids.size(); ++i) {
        std::cout << "Triangle " << i << " originates from input triangle "
                  << triangle_source_ids[i] << '\n';
    }

    Eigen::VectorXi triangle_origin_ids = Eigen::VectorXi::Constant(triangles.size(), -1);
    const std::size_t invalid_id = static_cast<std::size_t>(-1);
    for (Eigen::Index i = 0; i < triangle_origin_ids.size() && i < static_cast<Eigen::Index>(triangle_source_ids.size()); ++i) {
        const std::size_t src_id = triangle_source_ids[i];
        if (src_id == invalid_id) {
            continue;
        }
        if (src_id > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            std::cerr << "Warning: source triangle id " << src_id
                      << " exceeds Int32 range; writing as -1 in VTU.\n";
            continue;
        }
        triangle_origin_ids(static_cast<int>(i)) = static_cast<int>(src_id);
    }

    const Eigen::MatrixXd V_before = to_vertex_matrix(original_points);
    const Eigen::MatrixXi F_before = to_face_matrix(original_triangles);
    const Eigen::MatrixXd V_after = to_vertex_matrix(points);
    const Eigen::MatrixXi F_after = to_face_matrix(triangles);

    const std::string before_path = "cgal_autorefine_before.vtu";
    const std::string after_path = "cgal_autorefine_after.vtu";

    vtu_utils::write_triangle_mesh_to_vtu(V_before, F_before, before_path);
    vtu_utils::write_triangle_mesh_to_vtu(
        V_after,
        F_after,
        after_path,
        triangle_origin_ids.size() == F_after.rows() ? &triangle_origin_ids : nullptr,
        "origin_triangle_id");

    std::cout << "Wrote VTU snapshots:\n"
              << "  initial soup -> " << before_path << '\n'
              << "  refined soup -> " << after_path << '\n';

    return intersection_free ? 0 : 1;
}

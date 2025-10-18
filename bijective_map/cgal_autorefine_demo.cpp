#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/number_utils.h>
#include "vtu_utils.hpp"

#include <Eigen/Core>
#include <array>
#include <iostream>
#include <string>
#include <vector>

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

    namespace PMP = CGAL::Polygon_mesh_processing;

    const bool had_initial_intersections =
        PMP::does_triangle_soup_self_intersect(points, triangles);

    PMP::autorefine_triangle_soup(points, triangles, CGAL::parameters::default_values());

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

    const Eigen::MatrixXd V_before = to_vertex_matrix(original_points);
    const Eigen::MatrixXi F_before = to_face_matrix(original_triangles);
    const Eigen::MatrixXd V_after = to_vertex_matrix(points);
    const Eigen::MatrixXi F_after = to_face_matrix(triangles);

    const std::string before_path = "cgal_autorefine_before.vtu";
    const std::string after_path = "cgal_autorefine_after.vtu";

    vtu_utils::write_triangle_mesh_to_vtu(V_before, F_before, before_path);
    vtu_utils::write_triangle_mesh_to_vtu(V_after, F_after, after_path);

    std::cout << "Wrote VTU snapshots:\n"
              << "  initial soup -> " << before_path << '\n'
              << "  refined soup -> " << after_path << '\n';

    return intersection_free ? 0 : 1;
}

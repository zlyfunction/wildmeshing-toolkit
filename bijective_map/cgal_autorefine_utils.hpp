#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <Eigen/Core>

#include <array>
#include <set>
#include <vector>

namespace cgal_autorefine_demo {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Triangle = std::array<std::size_t, 3>;

struct TetTriangle
{
    Triangle triangle;
    std::vector<Eigen::Index> tet_indices;
};

std::vector<TetTriangle> extract_all_tet_triangles(const Eigen::MatrixXi& T);

struct SampledVertex
{
    std::size_t point_index;
    Eigen::Vector4d barycentric;
    Eigen::Vector3d position;
    Eigen::Index tet_index;
};

struct SampledPointInput
{
    Eigen::Index tet_index;
    Eigen::Vector4d barycentric;
};

struct AutorefineResult
{
    // Triangle soup immediately before CGAL autorefinement (tet boundary + test triangles).
    std::vector<Point> original_points;
    std::vector<Triangle> original_triangles;
    std::vector<std::vector<int>> original_triangle_parent_tets;

    // Triangle soup returned by CGAL autorefinement.
    std::vector<Point> refined_points;
    std::vector<Triangle> refined_triangles;

    // Per-refined-triangle provenance back to input triangle index and owning tet.
    Eigen::VectorXi origin_triangle_ids;
    Eigen::VectorXi origin_tet_ids;

    // Triangles produced from sampled test triangles, with their refined indices/source faces.
    std::vector<std::size_t> sampled_fragment_indices;
    std::vector<Triangle> sampled_fragment_triangles;
    Eigen::VectorXi sampled_fragment_tet_ids;
    std::vector<int> sampled_fragment_source_ids;

    // Sampled vertices with their source tetrahedra, plus per-vertex tet accumulator sets.
    std::vector<SampledVertex> sampled_vertices;
    std::vector<std::set<int>> vertex_tet_sets;

    // Intersection status before and after CGAL autorefinement.
    bool initial_soup_had_intersections = false;
    bool refined_soup_is_intersection_free = false;
};

AutorefineResult autorefine_sampled_triangles(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T,
    const std::vector<SampledPointInput>& sampled_points,
    const Eigen::MatrixXi& sampled_faces);

} // namespace cgal_autorefine_demo

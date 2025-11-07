#pragma once

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/NT_converter.h>
#include <wmtk/utils/Rational.hpp>

#include <Eigen/Core>

#include <array>
#include <set>
#include <vector>

namespace cgal_autorefine_demo {

using Triangle = std::array<std::size_t, 3>;

// Rational kernel for exact arithmetic
// Exact_predicates_exact_constructions_kernel uses Lazy_exact_nt<Gmpq> internally
using RationalKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using RationalPoint = RationalKernel::Point_3;

// Rational version structures
struct SampledVertexRational
{
    std::size_t point_index;
    Eigen::Matrix<wmtk::Rational, 4, 1> barycentric;
    Eigen::Matrix<wmtk::Rational, 3, 1> position;
    Eigen::Index tet_index;
};

struct SampledPointInputRational
{
    Eigen::Index tet_index;
    Eigen::Matrix<wmtk::Rational, 4, 1> barycentric;
};

struct AutorefineResultRational
{
    // Triangle soup immediately before CGAL autorefinement (tet boundary + test triangles).
    std::vector<RationalPoint> original_points;
    std::vector<Triangle> original_triangles;
    std::vector<std::vector<int>> original_triangle_parent_tets;

    // Triangle soup returned by CGAL autorefinement.
    std::vector<RationalPoint> refined_points;
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
    std::vector<SampledVertexRational> sampled_vertices;
    std::vector<std::set<int>> vertex_tet_sets;

    // Intersection status before and after CGAL autorefinement.
    bool initial_soup_had_intersections = false;
    bool refined_soup_is_intersection_free = false;
};

AutorefineResultRational autorefine_sampled_triangles_rational(
    const Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, 3>& V,
    const Eigen::MatrixXi& T,
    const std::vector<SampledPointInputRational>& sampled_points,
    const Eigen::MatrixXi& sampled_faces);

} // namespace cgal_autorefine_demo


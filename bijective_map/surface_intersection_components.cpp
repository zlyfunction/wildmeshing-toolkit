#include "surface_intersection_components.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/intersections.h>
#include <gmp.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tet_surface_tracking_with_connectivity {
namespace {
using RationalKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using RationalPoint = RationalKernel::Point_3;
using RationalSegment = RationalKernel::Segment_3;
using RationalTriangle = RationalKernel::Triangle_3;

RationalKernel::FT rational_to_gmpq(const wmtk::Rational& r)
{
    mpq_t q;
    mpq_init(q);
    r.export_mpq(q);
    using ET = RationalKernel::FT::ET;
    ET et_expr(q);
    RationalKernel::FT result(et_expr);
    mpq_clear(q);
    return result;
}

struct PointLess
{
    bool operator()(const RationalPoint& a, const RationalPoint& b) const
    {
        static const RationalKernel kernel;
        const auto compare = kernel.compare_xyz_3_object();
        return compare(a, b) == CGAL::SMALLER;
    }
};

class DisjointSet
{
public:
    std::size_t add()
    {
        const std::size_t id = parent_.size();
        parent_.push_back(id);
        rank_.push_back(0);
        return id;
    }

    std::size_t find(std::size_t x)
    {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    void unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (rank_[a] < rank_[b]) {
            std::swap(a, b);
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) {
            rank_[a]++;
        }
    }

    std::size_t size() const { return parent_.size(); }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> rank_;
};

struct SegmentRecord
{
    RationalSegment segment;
    std::size_t a = 0;
    std::size_t b = 0;
};

std::vector<RationalPoint> build_world_points(
    const query_surface_tet_with_connectivity& surface,
    const MatrixXr& tet_vertices,
    const std::string& label)
{
    if (tet_vertices.cols() != 3) {
        throw std::runtime_error("tet_vertices must have 3 columns.");
    }
    std::vector<RationalPoint> points;
    points.reserve(surface.points.size());
    for (std::size_t i = 0; i < surface.points.size(); ++i) {
        const auto& pt = surface.points[i];
        Eigen::Matrix<wmtk::Rational, 3, 1> world_pos =
            Eigen::Matrix<wmtk::Rational, 3, 1>::Zero();
        for (int j = 0; j < 4; ++j) {
            const int v_id = pt.tv_ids(j);
            if (v_id < 0 || v_id >= tet_vertices.rows()) {
                std::ostringstream oss;
                oss << "Invalid tv_ids entry for surface " << label << " point " << i
                    << ": tv_ids[" << j << "]=" << v_id
                    << " (V rows=" << tet_vertices.rows() << ")";
                throw std::runtime_error(oss.str());
            }
            world_pos += pt.bc(j) * tet_vertices.row(v_id).transpose();
        }
        const RationalKernel::FT x = rational_to_gmpq(world_pos(0));
        const RationalKernel::FT y = rational_to_gmpq(world_pos(1));
        const RationalKernel::FT z = rational_to_gmpq(world_pos(2));
        points.emplace_back(x, y, z);
    }
    return points;
}

RationalTriangle make_triangle(
    const std::vector<RationalPoint>& points,
    const Eigen::Vector3i& tri,
    const std::string& label,
    std::size_t tri_index)
{
    const auto check_idx = [&](int idx) {
        if (idx < 0 || static_cast<std::size_t>(idx) >= points.size()) {
            std::ostringstream oss;
            oss << "Triangle index out of range for surface " << label << " triangle "
                << tri_index << ": " << idx << " (point count=" << points.size() << ")";
            throw std::runtime_error(oss.str());
        }
    };
    check_idx(tri(0));
    check_idx(tri(1));
    check_idx(tri(2));
    if (tri(0) == tri(1) || tri(1) == tri(2) || tri(0) == tri(2)) {
        std::ostringstream oss;
        oss << "Degenerate triangle in surface " << label << " triangle " << tri_index
            << " indices=(" << tri(0) << "," << tri(1) << "," << tri(2) << ")";
        throw std::runtime_error(oss.str());
    }
    return RationalTriangle(points[tri(0)], points[tri(1)], points[tri(2)]);
}

std::size_t get_point_id(
    const RationalPoint& p,
    std::map<RationalPoint, std::size_t, PointLess>& point_ids,
    std::vector<RationalPoint>& id_to_point,
    DisjointSet& dsu)
{
    auto it = point_ids.find(p);
    if (it != point_ids.end()) {
        return it->second;
    }
    const std::size_t id = dsu.add();
    point_ids.emplace(p, id);
    id_to_point.push_back(p);
    return id;
}
} // namespace

IntersectionComponentResult compute_surface_intersection_components(
    const query_surface_tet_with_connectivity& surface_a,
    const query_surface_tet_with_connectivity& surface_b,
    const MatrixXr& tet_vertices,
    bool verbose)
{
    const std::vector<RationalPoint> points_a = build_world_points(surface_a, tet_vertices, "A");
    const std::vector<RationalPoint> points_b = build_world_points(surface_b, tet_vertices, "B");
    std::vector<RationalTriangle> triangles_a;
    std::vector<RationalTriangle> triangles_b;
    triangles_a.reserve(surface_a.query_triangles.size());
    triangles_b.reserve(surface_b.query_triangles.size());
    for (std::size_t i = 0; i < surface_a.query_triangles.size(); ++i) {
        triangles_a.push_back(make_triangle(points_a, surface_a.query_triangles[i], "A", i));
    }
    for (std::size_t j = 0; j < surface_b.query_triangles.size(); ++j) {
        triangles_b.push_back(make_triangle(points_b, surface_b.query_triangles[j], "B", j));
    }

    std::map<RationalPoint, std::size_t, PointLess> point_ids;
    std::vector<RationalPoint> id_to_point;
    DisjointSet dsu;
    std::vector<SegmentRecord> segments;

    for (std::size_t i = 0; i < triangles_a.size(); ++i) {
        const auto& tri_a = triangles_a[i];
        for (std::size_t j = 0; j < triangles_b.size(); ++j) {
            const auto& tri_b = triangles_b[j];
            const auto result = CGAL::intersection(tri_a, tri_b);
            if (!result) {
                continue;
            }
            const auto& intersection = *result;
            if (const auto* seg = std::get_if<RationalSegment>(&intersection)) {
                if (seg->source() == seg->target()) {
                    get_point_id(seg->source(), point_ids, id_to_point, dsu);
                } else {
                    const std::size_t a =
                        get_point_id(seg->source(), point_ids, id_to_point, dsu);
                    const std::size_t b =
                        get_point_id(seg->target(), point_ids, id_to_point, dsu);
                    dsu.unite(a, b);
                    segments.push_back({*seg, a, b});
                }
                continue;
            }
            if (const auto* pt = std::get_if<RationalPoint>(&intersection)) {
                get_point_id(*pt, point_ids, id_to_point, dsu);
                continue;
            }
            if (const auto* tri = std::get_if<RationalTriangle>(&intersection)) {
                (void)tri;
                std::ostringstream oss;
                oss << "Coplanar overlap detected between surface triangles A[" << i << "] and B["
                    << j << "]";
                throw std::runtime_error(oss.str());
            }
            if (const auto* poly = std::get_if<std::vector<RationalPoint>>(&intersection)) {
                (void)poly;
                std::ostringstream oss;
                oss << "Coplanar overlap detected between surface triangles A[" << i << "] and B["
                    << j << "]";
                throw std::runtime_error(oss.str());
            }
            std::ostringstream oss;
            oss << "Unexpected triangle intersection type for A[" << i << "] and B[" << j << "]";
            throw std::runtime_error(oss.str());
        }
    }

    if (!segments.empty() && !id_to_point.empty()) {
        for (const auto& seg : segments) {
            for (std::size_t pid = 0; pid < id_to_point.size(); ++pid) {
                if (seg.segment.has_on(id_to_point[pid])) {
                    dsu.unite(seg.a, pid);
                    dsu.unite(seg.b, pid);
                }
            }
        }
    }

    IntersectionComponentResult result;
    result.segment_count = segments.size();
    result.point_count = id_to_point.size();
    if (dsu.size() == 0) {
        result.component_count = 0;
    } else {
        std::set<std::size_t> roots;
        for (std::size_t i = 0; i < dsu.size(); ++i) {
            roots.insert(dsu.find(i));
        }
        result.component_count = roots.size();
    }

    if (verbose) {
        std::cout << "Intersection summary: components=" << result.component_count
                  << " segments=" << result.segment_count << " points=" << result.point_count
                  << std::endl;
    }

    return result;
}

} // namespace tet_surface_tracking_with_connectivity

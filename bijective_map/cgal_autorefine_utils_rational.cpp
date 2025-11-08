#include "cgal_autorefine_utils_rational.hpp"
#include "cgal_autorefine_utils.hpp"

#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/number_utils.h>
#include <gmp.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;

namespace cgal_autorefine_demo {

namespace {

struct TriangleTrackingVisitorRational : PMP::Autorefinement::Default_visitor
{
    TriangleTrackingVisitorRational() = default;
    explicit TriangleTrackingVisitorRational(std::vector<std::size_t>& mapping)
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

    void new_subtriangle(std::size_t tgt_id, std::size_t src_id) { store_mapping(tgt_id, src_id); }

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

// Convert wmtk::Rational to RationalKernel::FT (Lazy_exact_nt<Gmpq>)
RationalKernel::FT rational_to_gmpq(const wmtk::Rational& r)
{
    mpq_t q;
    mpq_init(q);
    r.export_mpq(q);
    // Lazy_exact_nt<Gmpq> constructor takes const ET& where ET is __gmp_expr<mpq_t, mpq_t>
    // We can construct this from mpq_t using the explicit constructor
    using ET = RationalKernel::FT::ET;
    ET et_expr(q); // __gmp_expr has explicit constructor from mpq_srcptr
    RationalKernel::FT result(et_expr);
    mpq_clear(q);
    return result;
}

std::vector<RationalPoint> rational_vertices_to_points(
    const Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, 3>& V)
{
    std::vector<RationalPoint> points;
    points.reserve(static_cast<std::size_t>(V.rows()));
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        RationalKernel::FT x = rational_to_gmpq(V(i, 0));
        RationalKernel::FT y = rational_to_gmpq(V(i, 1));
        RationalKernel::FT z = rational_to_gmpq(V(i, 2));
        points.emplace_back(x, y, z);
    }
    return points;
}

} // namespace

AutorefineResultRational autorefine_sampled_triangles_rational(
    const Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, 3>& V,
    const Eigen::MatrixXi& T,
    const std::vector<SampledPointInputRational>& sampled_points,
    const Eigen::MatrixXi& sampled_faces)
{
    AutorefineResultRational result;

    std::vector<RationalPoint> points = rational_vertices_to_points(V);
    std::vector<Triangle> triangles;
    triangles.reserve(
        static_cast<std::size_t>(T.rows() * 4) + static_cast<std::size_t>(sampled_faces.rows()));
    std::vector<std::vector<int>> triangle_parent_tets;
    triangle_parent_tets.reserve(triangles.capacity());
    std::vector<int> triangle_sample_ids;
    triangle_sample_ids.reserve(triangles.capacity());

    const auto tet_triangles = extract_all_tet_triangles(T);
    std::cout << "T:\n" << T << std::endl;
    std::cout << "tet_triangles size: " << tet_triangles.size() << std::endl;
    for (const TetTriangle& face : tet_triangles) {
        triangles.push_back(face.triangle);
        std::vector<int> parent_ids;
        parent_ids.reserve(face.tet_indices.size());
        for (Eigen::Index tet_id : face.tet_indices) {
            parent_ids.push_back(static_cast<int>(tet_id));
        }
        triangle_parent_tets.push_back(std::move(parent_ids));
        triangle_sample_ids.push_back(-1);
    }
    std::cout << "TET size: " << tet_triangles.size() << std::endl;
    std::cout << "triangles size: " << triangles.size() << std::endl;
    std::vector<SampledVertexRational> sampled_vertices;
    sampled_vertices.reserve(sampled_points.size());
    std::vector<std::size_t> sampled_point_global_indices;
    sampled_point_global_indices.reserve(sampled_points.size());

    for (const auto& point_input : sampled_points) {
        if (point_input.tet_index < 0 || point_input.tet_index >= T.rows()) {
            throw std::runtime_error("Sampled point references invalid tetrahedron index.");
        }
        Eigen::Matrix<wmtk::Rational, 3, 4> tet_vertices;
        for (int i = 0; i < 4; ++i) {
            const Eigen::Index v_idx = T(point_input.tet_index, i);
            tet_vertices.col(i) = V.row(v_idx);
        }

        const Eigen::Matrix<wmtk::Rational, 4, 1>& bc = point_input.barycentric;
        Eigen::Matrix<wmtk::Rational, 3, 1> position =
            bc[0] * tet_vertices.col(0) + bc[1] * tet_vertices.col(1) +
            bc[2] * tet_vertices.col(2) + bc[3] * tet_vertices.col(3);

        std::size_t point_index = points.size();
        RationalKernel::FT x = rational_to_gmpq(position(0));
        RationalKernel::FT y = rational_to_gmpq(position(1));
        RationalKernel::FT z = rational_to_gmpq(position(2));
        points.emplace_back(x, y, z);
        std::cout << "Sampled Rational Point: barycentric = [";
        for (int j = 0; j < 4; ++j) {
            std::cout << bc[j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << "], tet_index = " << point_input.tet_index << ", position = [";
        for (int j = 0; j < 3; ++j) {
            std::cout << position(j);
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        SampledVertexRational vertex;
        vertex.point_index = point_index;
        vertex.barycentric = bc;
        vertex.position = position;
        vertex.tet_index = point_input.tet_index;
        sampled_vertices.push_back(vertex);
        sampled_point_global_indices.push_back(point_index);
    }

    for (Eigen::Index face_id = 0; face_id < sampled_faces.rows(); ++face_id) {
        Triangle tri{};
        for (int corner = 0; corner < 3; ++corner) {
            const int local_index = sampled_faces(face_id, corner);
            if (local_index < 0 ||
                local_index >= static_cast<int>(sampled_point_global_indices.size())) {
                throw std::runtime_error("Sampled triangle references invalid vertex index.");
            }
            const std::size_t global_index =
                sampled_point_global_indices[static_cast<std::size_t>(local_index)];
            tri[corner] = global_index;
        }
        triangles.push_back(tri);
        triangle_parent_tets.emplace_back();
        triangle_sample_ids.push_back(static_cast<int>(face_id));
    }

    result.original_points = points;
    result.original_triangles = triangles;
    result.original_triangle_parent_tets = triangle_parent_tets;

    std::vector<std::vector<std::size_t>> working_triangles;
    working_triangles.reserve(triangles.size());
    for (const Triangle& tri : triangles) {
        working_triangles.push_back({tri[0], tri[1], tri[2]});
    }

    result.initial_soup_had_intersections =
        PMP::does_triangle_soup_self_intersect(points, triangles);
    std::cout << "Points:\n";
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& pt = points[i];
        std::cout << "  " << i << ": [" << CGAL::to_double(pt.x()) << ", "
                  << CGAL::to_double(pt.y()) << ", " << CGAL::to_double(pt.z()) << "]\n";
    }
    std::cout << "Triangles:\n";
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        std::cout << "  " << i << ": [" << tri[0] << ", " << tri[1] << ", " << tri[2] << "]\n";
    }
    std::vector<std::size_t> triangle_source_ids;
    TriangleTrackingVisitorRational visitor(triangle_source_ids);
    PMP::autorefine_triangle_soup(
        points,
        working_triangles,
        CGAL::parameters::visitor(visitor).apply_iterative_snap_rounding(true));

    const std::size_t invalid_id = static_cast<std::size_t>(-1);

    triangles.clear();
    triangles.reserve(working_triangles.size());
    std::vector<std::size_t> filtered_source_ids;
    filtered_source_ids.reserve(working_triangles.size());
    std::vector<std::vector<int>> refined_triangle_parent_tets;
    refined_triangle_parent_tets.reserve(working_triangles.size());
    std::vector<int> refined_triangle_sample_ids;
    refined_triangle_sample_ids.reserve(working_triangles.size());

    for (std::size_t i = 0; i < working_triangles.size(); ++i) {
        const auto& tri = working_triangles[i];
        if (tri.size() != 3) {
            std::cerr << "Warning: encountered triangle with " << tri.size()
                      << " vertices after autorefinement; skipping.\n";
            continue;
        }

        triangles.push_back(Triangle{tri[0], tri[1], tri[2]});

        const std::size_t src_id =
            (i < triangle_source_ids.size()) ? triangle_source_ids[i] : invalid_id;
        filtered_source_ids.push_back(src_id);

        if (src_id != invalid_id && src_id < triangle_parent_tets.size()) {
            refined_triangle_parent_tets.push_back(triangle_parent_tets[src_id]);
        } else {
            refined_triangle_parent_tets.emplace_back();
        }

        int sample_id = -1;
        if (src_id != invalid_id && src_id < triangle_sample_ids.size()) {
            sample_id = triangle_sample_ids[src_id];
        }
        refined_triangle_sample_ids.push_back(sample_id);
    }
    triangle_source_ids.swap(filtered_source_ids);

    result.refined_soup_is_intersection_free =
        !PMP::does_triangle_soup_self_intersect(points, triangles);

    Eigen::VectorXi origin_triangle_ids = Eigen::VectorXi::Constant(triangles.size(), -1);
    Eigen::VectorXi origin_tet_ids = Eigen::VectorXi::Constant(triangles.size(), -1);

    for (Eigen::Index i = 0; i < origin_triangle_ids.size(); ++i) {
        if (i < static_cast<Eigen::Index>(triangle_source_ids.size())) {
            const std::size_t src_id = triangle_source_ids[static_cast<std::size_t>(i)];
            if (src_id != invalid_id &&
                src_id <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                origin_triangle_ids(i) = static_cast<int>(src_id);
            }
        }
        if (i < static_cast<Eigen::Index>(refined_triangle_parent_tets.size())) {
            const auto& parents = refined_triangle_parent_tets[static_cast<std::size_t>(i)];
            if (!parents.empty()) {
                origin_tet_ids(i) = parents.front();
            }
        }
    }

    std::vector<std::set<int>> vertex_tet_sets(points.size());
    for (const SampledVertexRational& vertex : sampled_vertices) {
        if (vertex.point_index < vertex_tet_sets.size()) {
            vertex_tet_sets[vertex.point_index].insert(static_cast<int>(vertex.tet_index));
        }
    }

    for (std::size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
        if (tri_idx >= refined_triangle_parent_tets.size()) {
            continue;
        }
        if (tri_idx < refined_triangle_sample_ids.size() &&
            refined_triangle_sample_ids[tri_idx] >= 0) {
            continue;
        }
        const Triangle& tri = triangles[tri_idx];
        const auto& parents = refined_triangle_parent_tets[tri_idx];
        for (int tet_id : parents) {
            if (tet_id < 0) {
                continue;
            }
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const std::size_t v_id = tri[corner];
                if (v_id < vertex_tet_sets.size()) {
                    vertex_tet_sets[v_id].insert(tet_id);
                }
            }
        }
    }

    std::vector<Triangle> sampled_fragment_triangles;
    std::vector<std::size_t> sampled_fragment_indices;
    std::vector<int> sampled_fragment_source_ids;
    std::vector<int> sampled_fragment_tet_list;

    for (std::size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
        const int sample_id = refined_triangle_sample_ids[tri_idx];
        if (sample_id < 0) {
            continue;
        }

        sampled_fragment_triangles.push_back(triangles[tri_idx]);
        sampled_fragment_indices.push_back(tri_idx);
        sampled_fragment_source_ids.push_back(sample_id);

        std::set<int> common_tets = vertex_tet_sets[triangles[tri_idx][0]];
        for (int corner = 1; corner < 3 && !common_tets.empty(); ++corner) {
            std::set<int> temp;
            const std::set<int>& tet_set =
                vertex_tet_sets[triangles[tri_idx][static_cast<std::size_t>(corner)]];
            std::set_intersection(
                common_tets.begin(),
                common_tets.end(),
                tet_set.begin(),
                tet_set.end(),
                std::inserter(temp, temp.begin()));
            common_tets.swap(temp);
        }

        const int assigned_tet = common_tets.empty() ? -1 : *common_tets.begin();
        sampled_fragment_tet_list.push_back(assigned_tet);
        if (assigned_tet != -1 && tri_idx < static_cast<std::size_t>(origin_tet_ids.size())) {
            origin_tet_ids(static_cast<Eigen::Index>(tri_idx)) = assigned_tet;
        }
    }

    for (SampledVertexRational& vertex : sampled_vertices) {
        if (vertex.point_index < points.size()) {
            const RationalPoint& p = points[vertex.point_index];
            vertex.position(0) = wmtk::Rational(CGAL::to_double(p.x()));
            vertex.position(1) = wmtk::Rational(CGAL::to_double(p.y()));
            vertex.position(2) = wmtk::Rational(CGAL::to_double(p.z()));
        }
    }

    Eigen::VectorXi sampled_fragment_tet_ids(sampled_fragment_tet_list.size());
    for (Eigen::Index i = 0; i < sampled_fragment_tet_ids.size(); ++i) {
        sampled_fragment_tet_ids(i) = sampled_fragment_tet_list[static_cast<std::size_t>(i)];
    }

    result.refined_points = points;
    result.refined_triangles = triangles;
    result.origin_triangle_ids = std::move(origin_triangle_ids);
    result.origin_tet_ids = std::move(origin_tet_ids);
    result.sampled_fragment_indices = std::move(sampled_fragment_indices);
    result.sampled_fragment_triangles = std::move(sampled_fragment_triangles);
    result.sampled_fragment_tet_ids = std::move(sampled_fragment_tet_ids);
    result.sampled_fragment_source_ids = std::move(sampled_fragment_source_ids);
    result.sampled_vertices = std::move(sampled_vertices);
    result.vertex_tet_sets = std::move(vertex_tet_sets);

    return result;
}

} // namespace cgal_autorefine_demo

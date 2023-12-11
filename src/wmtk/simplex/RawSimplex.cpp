#include "RawSimplex.hpp"

#include <algorithm>

#include "RawSimplexCollection.hpp"
#include "Simplex.hpp"
#include "faces_single_dimension.hpp"

namespace wmtk::simplex {
RawSimplex::RawSimplex(const Mesh& mesh, const std::vector<Tuple>& vertices)
{
    m_vertices.reserve(vertices.size());

    ConstAccessor<long> hash_accessor = mesh.get_const_cell_hash_accessor();

    for (size_t i = 0; i < vertices.size(); ++i) {
        m_vertices.emplace_back(
            mesh.is_valid(vertices[i], hash_accessor) ? mesh.id(vertices[i], PrimitiveType::Vertex)
                                                      : -1);
    }

    std::sort(m_vertices.begin(), m_vertices.end());
}

RawSimplex::RawSimplex(std::vector<long>&& vertices)
    : m_vertices{std::move(vertices)}
{
    std::sort(m_vertices.begin(), m_vertices.end());
}

RawSimplex::RawSimplex(const Mesh& mesh, const Simplex& simplex)
    : RawSimplex(
          mesh,
          simplex.primitive_type() == PrimitiveType::Vertex
              ? std::vector<Tuple>{simplex.tuple()}
              : faces_single_dimension_tuples(mesh, simplex, PrimitiveType::Vertex))
{}

long RawSimplex::dimension() const
{
    return m_vertices.size() - 1;
}

bool RawSimplex::operator==(const RawSimplex& o) const
{
    return std::equal(
        m_vertices.begin(),
        m_vertices.end(),
        o.m_vertices.begin(),
        o.m_vertices.end());
}

bool RawSimplex::operator<(const RawSimplex& o) const
{
    if (dimension() != o.dimension()) {
        return dimension() < o.dimension();
    }

    return std::lexicographical_compare(
        m_vertices.begin(),
        m_vertices.end(),
        o.m_vertices.begin(),
        o.m_vertices.end());
}

RawSimplex RawSimplex::opposite_face(const long excluded_id)
{
    std::vector<long> face_ids;
    face_ids.reserve(m_vertices.size() - 1);

    for (const long& v : m_vertices) {
        if (v != excluded_id) {
            face_ids.emplace_back(v);
        }
    }

    RawSimplex face(std::move(face_ids));
    assert(face.dimension() == dimension() - 1);

    return face;
}

RawSimplex RawSimplex::opposite_face(const Mesh& mesh, const Tuple& vertex)
{
    ConstAccessor<long> hash_accessor = mesh.get_const_cell_hash_accessor();

    long excluded_id =
        mesh.is_valid(vertex, hash_accessor) ? mesh.id(vertex, PrimitiveType::Vertex) : -1;

    return opposite_face(excluded_id);
}

RawSimplexCollection RawSimplex::faces()
{
    const size_t& nv = m_vertices.size();
    const auto& v = m_vertices;

    std::vector<RawSimplex> faces;

    switch (dimension()) {
    case 1: { // simplex is an edge
        faces.reserve(2);
        faces.emplace_back(RawSimplex({v[0]}));
        faces.emplace_back(RawSimplex({v[1]}));
        break;
    }
    case 2: { // simplex is a triangle
        faces.reserve(6);
        faces.emplace_back(RawSimplex({v[0]}));
        faces.emplace_back(RawSimplex({v[1]}));
        faces.emplace_back(RawSimplex({v[2]}));
        faces.emplace_back(RawSimplex({v[0], v[1]}));
        faces.emplace_back(RawSimplex({v[0], v[2]}));
        faces.emplace_back(RawSimplex({v[1], v[2]}));
        break;
    }
    case 3: { // simplex is a tetrahedron
        faces.reserve(14);
        faces.emplace_back(RawSimplex({v[0]}));
        faces.emplace_back(RawSimplex({v[1]}));
        faces.emplace_back(RawSimplex({v[2]}));
        faces.emplace_back(RawSimplex({v[3]}));
        faces.emplace_back(RawSimplex({v[0], v[1]}));
        faces.emplace_back(RawSimplex({v[0], v[2]}));
        faces.emplace_back(RawSimplex({v[0], v[3]}));
        faces.emplace_back(RawSimplex({v[1], v[2]}));
        faces.emplace_back(RawSimplex({v[1], v[3]}));
        faces.emplace_back(RawSimplex({v[2], v[3]}));
        faces.emplace_back(RawSimplex({v[0], v[1], v[2]}));
        faces.emplace_back(RawSimplex({v[0], v[1], v[3]}));
        faces.emplace_back(RawSimplex({v[0], v[2], v[3]}));
        faces.emplace_back(RawSimplex({v[1], v[2], v[3]}));
        break;
    }
    default: throw std::runtime_error("Unexpected dimension in RawSimplex."); break;
    }

    return RawSimplexCollection(std::move(faces));
}

} // namespace wmtk::simplex
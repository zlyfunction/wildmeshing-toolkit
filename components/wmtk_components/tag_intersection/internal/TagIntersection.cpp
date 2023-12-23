#include "TagIntersection.hpp"

namespace wmtk {
namespace components {

bool TagIntersection::simplex_is_in_intersection(
    Mesh& m,
    const Simplex& v,
    const std::deque<TagAttribute>& input_tag_attributes)
{
    const simplex::SimplexCollection os = simplex::open_star(m, v);

    std::vector<bool> tag_is_present(input_tag_attributes.size(), false);

    for (const Simplex& s : os.simplex_vector()) {
        for (size_t i = 0; i < input_tag_attributes.size(); ++i) {
            if (input_tag_attributes[i].is_tagged(m, s)) {
                tag_is_present[i] = true;
            }
        }
    }

    // check if all tags are present
    bool is_intersection = true;
    for (const bool b : tag_is_present) {
        is_intersection = is_intersection && b;
    }
    return is_intersection;
}


void TagIntersection::compute_intersection(
    TriMesh& m,
    const std::vector<std::tuple<MeshAttributeHandle<long>, long>>& input_tags,
    const std::vector<std::tuple<MeshAttributeHandle<long>, long>>& output_tags)
{
    std::deque<TagAttribute> input_tag_attributes;
    for (const auto& [handle, val] : input_tags) {
        input_tag_attributes.emplace_back(m, handle, handle.primitive_type(), val);
    }
    std::deque<TagAttribute> output_tag_attributes;
    for (const auto& [handle, val] : output_tags) {
        output_tag_attributes.emplace_back(m, handle, handle.primitive_type(), val);
    }

    // iterate through all simplices and check if the simplex itself or any of its cofaces is
    // tagged
    for (const Tuple& t : m.get_all(PrimitiveType::Vertex)) {
        const Simplex v = Simplex::vertex(t);
        if (simplex_is_in_intersection(m, v, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, v);
            }
        }
    }
    for (const Tuple& t : m.get_all(PrimitiveType::Edge)) {
        const Simplex e = Simplex::edge(t);
        if (simplex_is_in_intersection(m, e, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, e);
            }
        }
    }
    for (const Tuple& t : m.get_all(PrimitiveType::Face)) {
        const Simplex f = Simplex::face(t);
        if (simplex_is_in_intersection(m, f, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, f);
            }
        }
    }
}

void TagIntersection::compute_intersection(
    TetMesh& m,
    const std::vector<std::tuple<MeshAttributeHandle<long>, long>>& input_tags,
    const std::vector<std::tuple<MeshAttributeHandle<long>, long>>& output_tags)
{
    std::deque<TagAttribute> input_tag_attributes;
    for (const auto& [handle, val] : input_tags) {
        input_tag_attributes.emplace_back(m, handle, handle.primitive_type(), val);
    }
    std::deque<TagAttribute> output_tag_attributes;
    for (const auto& [handle, val] : output_tags) {
        output_tag_attributes.emplace_back(m, handle, handle.primitive_type(), val);
    }

    for (const Tuple& t : m.get_all(PrimitiveType::Vertex)) {
        const Simplex v = Simplex::vertex(t);
        if (simplex_is_in_intersection(m, v, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, v);
            }
        }
    }
    for (const Tuple& t : m.get_all(PrimitiveType::Edge)) {
        const Simplex e = Simplex::edge(t);
        if (simplex_is_in_intersection(m, e, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, e);
            }
        }
    }
    for (const Tuple& t : m.get_all(PrimitiveType::Face)) {
        const Simplex f = Simplex::face(t);
        if (simplex_is_in_intersection(m, f, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, f);
            }
        }
    }
    for (const Tuple& t : m.get_all(PrimitiveType::Tetrahedron)) {
        const Simplex f = Simplex::tetrahedron(t);
        if (simplex_is_in_intersection(m, f, input_tag_attributes)) {
            for (TagAttribute& ta : output_tag_attributes) {
                ta.set_tag(m, f);
            }
        }
    }
}

} // namespace components
} // namespace wmtk

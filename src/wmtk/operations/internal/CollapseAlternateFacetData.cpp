#include "CollapseAlternateFacetData.hpp"
#include <spdlog/spdlog.h>
#include <array>
#include <vector>
#include <wmtk/Mesh.hpp>
#include <wmtk/autogen/Dart.hpp>
#include <wmtk/autogen/SimplexDart.hpp>
#include <wmtk/autogen/local_dart_action.hpp>
#include <wmtk/multimesh/utils/find_local_dart_action.hpp>
#include <wmtk/multimesh/utils/find_local_switch_sequence.hpp>
#include <wmtk/utils/primitive_range.hpp>
#include "wmtk/autogen/SimplexDart.hpp"
#include "wmtk/utils/TupleInspector.hpp"

namespace wmtk::operations::internal {

void CollapseAlternateFacetData::add(const Mesh& m, const Tuple& input_tuple)
{
    m_data.emplace_back(m, input_tuple);
    // first tuple is different from the input by switching everything but vertex
    // second one is switch everything
}


CollapseAlternateFacetData::CollapseAlternateFacetData() = default;
CollapseAlternateFacetData::~CollapseAlternateFacetData() = default;

auto CollapseAlternateFacetData::get_alternative_data_it(const int64_t& input_facet) const
    -> AltData::const_iterator
{
    constexpr auto sort_op = [](const Data& value, const int64_t& facet_id) -> bool {
        return value.input.global_id() < facet_id;
    };
    auto it = std::lower_bound(m_data.begin(), m_data.end(), input_facet, sort_op);
    auto end = m_data.cend();

    // if we found

    // fix case where the lower bound was not the target value
    if (it != end && it->input.global_id() != input_facet) {
        it = end;
    }
    return it;
}
auto CollapseAlternateFacetData::get_alternatives_data(const Tuple& t) const -> const Data&
{
    auto it = get_alternative_data_it(t.m_global_cid);
    assert(it != m_data.cend());
    return *it;
}
std::array<Tuple, 2> CollapseAlternateFacetData::get_alternatives(
    const PrimitiveType mesh_pt,
    const Tuple& t) const
{
    const auto& data = get_alternatives_data(t);

    const wmtk::autogen::SimplexDart& sd = wmtk::autogen::SimplexDart::get_singleton(mesh_pt);
    const wmtk::autogen::Dart t_dart = sd.dart_from_tuple(t);

    const int8_t action =
        wmtk::multimesh::utils::find_local_dart_action(mesh_pt, t_dart, data.input);
    auto map = [action, &sd](const wmtk::autogen::Dart& tup) -> Tuple {
        if (tup.is_null()) {
            return {};
        } else {
            return sd.tuple_from_dart(wmtk::autogen::local_dart_action(sd, tup, action));
        }
    };

    std::array<Tuple, 2> r{{map(data.alts[0]), map(data.alts[1])}};

    return r;
}
Tuple CollapseAlternateFacetData::get_alternative(const PrimitiveType pt, const Tuple& t) const
{
    auto alts = get_alternatives(pt, t);
    assert(!alts[0].is_null() || !alts[1].is_null());
    if (!alts[0].is_null()) {
        return alts[0];
    } else {
        return alts[1];
    }
    return {};
    //
}
} // namespace wmtk::operations::internal

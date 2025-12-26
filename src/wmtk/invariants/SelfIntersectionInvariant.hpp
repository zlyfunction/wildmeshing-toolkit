#pragma once

#include <wmtk/attribute/TypedAttributeHandle.hpp>
#include "Invariant.hpp"

namespace wmtk {
template <typename T>
class SelfIntersectionInvariant : public Invariant
{
public:
    SelfIntersectionInvariant(const Mesh& m, const TypedAttributeHandle<T>& coordinate);
    using Invariant::Invariant;

    bool after(const std::vector<Tuple>&, const std::vector<Tuple>& top_dimension_tuples_after)
        const override;

private:
    const TypedAttributeHandle<T> m_coordinate_handle;
};
} // namespace wmtk


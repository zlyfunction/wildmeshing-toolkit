#pragma once

#include <wmtk/operations/EdgeCollapse.hpp>
#include <wmtk/operations/EdgeSplit.hpp>

namespace wmtk::operations::composite {

/**
 * The return tuple is the new vertex, pointing to the original vertex.
 * This operation does not set vertex positions.
 *     / | \
 *    /  |  \
 *   /  _*_  \
 *  / _< f \_ \
 *  |/_______\|
 *   \       /
 *    \     /
 *     \   /
 **/
class TriFaceSplit : public Operation
{
public:
    TriFaceSplit(Mesh& m);

    PrimitiveType primitive_type() const override { return PrimitiveType::Face; }

    inline EdgeSplit& split() { return m_split; }
    inline EdgeCollapse& collapse() { return m_collapse; }

protected:
    std::vector<Simplex> unmodified_primitives(const Simplex& simplex) const override;
    std::vector<Simplex> execute(const Simplex& simplex) override;

private:
    EdgeSplit m_split;
    EdgeCollapse m_collapse;
};

} // namespace wmtk::operations::composite

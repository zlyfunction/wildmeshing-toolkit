#include <wmtk/TriMesh.h>
using namespace wmtk;
std::optional<std::vector<TriMesh::Tuple>> TriMesh::Operation::operator()(
    const TriMesh::Tuple& t,
    TriMesh& m)
{
#if defined(USE_OPERATION_LOGGER)
    // If the operation logger exists then log
    if (m.p_operation_logger) {
        auto& wp_op_rec = m.p_operation_recorder.local();
        wp_op_rec = m.p_operation_logger->start_ptr(m, name(), t.as_stl_array());
    }
#endif
    if (before_check(t, m)) {
        m.start_protected_connectivity();
        std::vector<Tuple> new_t = execute(t, m);

        m.start_protected_attributes();
        if (after_check(t, m, new_t) && invariants(t,m, new_t)) {
            m.release_protected_connectivity();
            m.release_protected_attributes();
            return new_t;
        } else {
#if defined(USE_OPERATION_LOGGER)
            if (auto& wp_op_rec = m.p_operation_recorder.local(); !wp_op_rec.expired()) {
                wp_op_rec.lock()->cancel();
            }
#endif

            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
        }
    }
    return {};
}

bool TriMesh::Operation::invariants(
    const TriMesh::Tuple&,
    TriMesh& m,
    const std::vector<TriMesh::Tuple>& new_tris) const
{
    return m.invariants(new_tris);
}

std::vector<TriMesh::Tuple> TriMesh::SplitEdge::execute(const TriMesh::Tuple& t, TriMesh& m)
{
    std::vector<TriMesh::Tuple> new_tris;
    m.split_edge_new(t, new_tris);
    return new_tris;
}
bool TriMesh::SplitEdge::before_check(const TriMesh::Tuple& t, TriMesh& m)
{
    return m.split_edge_before(t);
}
bool TriMesh::SplitEdge::after_check(
    const TriMesh::Tuple&,
    TriMesh& m,
    const std::vector<TriMesh::Tuple>& new_tris) const
{
    if(new_tris.size() == 0) { return false; }
    // if we see a tri, the 
    return m.split_edge_after(new_tris[0]);
}
std::string TriMesh::SplitEdge::name() const
{
    return "edge_split";
}


std::vector<TriMesh::Tuple> TriMesh::SwapEdge::execute(const TriMesh::Tuple& t, TriMesh& m)
{
    std::vector<TriMesh::Tuple> new_tris;
    m.swap_edge_new(t, new_tris);
    return new_tris;
}
bool TriMesh::SwapEdge::before_check(const TriMesh::Tuple& t, TriMesh& m)
{
    return m.swap_edge_before(t);
}
bool TriMesh::SwapEdge::after_check(
    const TriMesh::Tuple&,
    TriMesh& m,
    const std::vector<TriMesh::Tuple>& new_tris) const
{
    if(new_tris.size() == 0) { return false; }
    // if we see a tri, the 
    return m.swap_edge_after(new_tris[0]);
}
std::string TriMesh::SwapEdge::name() const
{
    return "edge_swap";
}


std::vector<TriMesh::Tuple> TriMesh::SmoothVertex::execute(const TriMesh::Tuple&, TriMesh&)
{
    return {};
}
bool TriMesh::SmoothVertex::before_check(const TriMesh::Tuple& t, TriMesh& m)
{
    return m.smooth_before(t);
}
bool TriMesh::SmoothVertex::after_check(
    const TriMesh::Tuple& t,
    TriMesh& m,
    const std::vector<TriMesh::Tuple>& ) const
{
    // if we see a tri, the 
    return m.smooth_after(t);
}
std::string TriMesh::SmoothVertex::name() const
{
    return "smooth_vertex";
}

bool TriMesh::SmoothVertex::invariants(
    const TriMesh::Tuple& t,
    TriMesh& m,
    const std::vector<TriMesh::Tuple>& ) const
{
    return m.invariants(m.get_one_ring_tris_for_vertex(t));
}


std::vector<TriMesh::Tuple> TriMesh::EdgeCollapse::execute(const TriMesh::Tuple&t, TriMesh&m)
{
    std::vector<TriMesh::Tuple> new_tris;
    m.collapse_edge_new(t, new_tris);
    return new_tris;
}
bool TriMesh::EdgeCollapse::before_check(const TriMesh::Tuple& t, TriMesh& m)
{
    return m.collapse_edge_before(t);
}
bool TriMesh::EdgeCollapse::after_check(
    const TriMesh::Tuple&t,
    TriMesh& m,
    const std::vector<TriMesh::Tuple>& ) const
{
    return m.collapse_edge_after(t);
}

std::string TriMesh::EdgeCollapse::name() const
{
    return "edge_collapse";
}

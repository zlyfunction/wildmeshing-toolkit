#include <wmtk/TriMeshOperation.h>
using namespace wmtk;
auto TriMeshOperation::operator()(const Tuple& t, TriMesh& m) -> ExecuteReturnData
{
    ExecuteReturnData retdata;
#if defined(USE_OPERATION_LOGGER)
    // If the operation logger exists then log
    if (m.p_operation_logger) {
        auto& wp_op_rec = m.p_operation_recorder.local();
        wp_op_rec = m.p_operation_logger->start_ptr(m, name(), t.as_stl_array());
    }
#endif
    if (before_check(t, m)) {
        m.start_protected_connectivity();
        retdata = execute(t, m);
        m.start_protected_attributes();

        if (after_check(retdata, m) && invariants(retdata, m)) {
            m.release_protected_connectivity();
            m.release_protected_attributes();
        } else {
#if defined(USE_OPERATION_LOGGER)
            if (auto& wp_op_rec = m.p_operation_recorder.local(); !wp_op_rec.expired()) {
                wp_op_rec.lock()->cancel();
            }
#endif

            retdata.success = false;
            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
        }
    }

    
    return retdata;
}

#if defined(USE_OPERATION_LOGGER)
std::weak_ptr<OperationRecorder> TriMeshOperation::recorder(TriMesh& m) const
{
    return m.p_operation_recorder.local();
}
#endif
bool TriMeshOperation::invariants(const ExecuteReturnData& ret_data, TriMesh& m)
{
    return m.invariants(ret_data.new_tris);
}

void TriMeshOperation::set_vertex_size(size_t v_cnt, TriMesh& m)
{
    m.current_vert_size = v_cnt;
    m.m_vertex_connectivity.m_attributes.resize(v_cnt);
    m.m_vertex_connectivity.shrink_to_fit();
    m.resize_mutex(m.vert_capacity());
}
void TriMeshOperation::set_tri_size(size_t t_cnt, TriMesh& m)
{
    m.current_tri_size = t_cnt;
    m.m_tri_connectivity.m_attributes.resize(t_cnt);
    m.m_tri_connectivity.shrink_to_fit();
}

auto TriMeshSplitEdgeOperation::execute(const Tuple& t, TriMesh& m) -> ExecuteReturnData
{
    std::vector<Tuple> new_tris;

    auto new_t = m.split_edge_new(t, new_tris);
    return {new_t, new_tris, true};
}
bool TriMeshSplitEdgeOperation::before_check(const Tuple& t, TriMesh& m)
{
    return m.split_edge_before(t);
}
bool TriMeshSplitEdgeOperation::after_check(const ExecuteReturnData& ret_data, TriMesh& m)
{
    return m.split_edge_after(ret_data.tuple);
}
std::string TriMeshSplitEdgeOperation::name() const
{
    return "edge_split";
}


auto TriMeshSwapEdgeOperation::execute(const Tuple& t, TriMesh& m) -> ExecuteReturnData
{
    std::vector<Tuple> new_tris;
    auto new_t = m.swap_edge_new(t, new_tris);
    return {new_t, new_tris, true};
}
bool TriMeshSwapEdgeOperation::before_check(const Tuple& t, TriMesh& m)
{
    return m.swap_edge_before(t);
}
bool TriMeshSwapEdgeOperation::after_check(const ExecuteReturnData& ret_data, TriMesh& m)
{
    return m.swap_edge_after(ret_data.tuple);
    ;
}
std::string TriMeshSwapEdgeOperation::name() const
{
    return "edge_swap";
}


auto TriMeshSmoothVertexOperation::execute(const Tuple& t, TriMesh&) -> ExecuteReturnData
{
    // always succeed and return the Tuple for the (vertex) that we pointed at
    return {t, {}, true};
}
bool TriMeshSmoothVertexOperation::before_check(const Tuple& t, TriMesh& m)
{
    return m.smooth_before(t);
}
bool TriMeshSmoothVertexOperation::after_check(const ExecuteReturnData& ret_data, TriMesh& m)
{
    // if we see a tri, the
    return m.smooth_after(ret_data.tuple);
}
std::string TriMeshSmoothVertexOperation::name() const
{
    return "vertex_smooth";
}

bool TriMeshSmoothVertexOperation::invariants(const ExecuteReturnData& ret_data, TriMesh& m)
{
    // our execute should have tuple set to the input tuple (vertex)
    return m.invariants(m.get_one_ring_tris_for_vertex(ret_data.tuple));
}


auto TriMeshEdgeCollapseOperation::execute(const Tuple& t, TriMesh& m) -> ExecuteReturnData
{
    std::vector<Tuple> new_tris;
    Tuple new_t = m.collapse_edge_new(t, new_tris);
    return {new_t, new_tris, true};
}
bool TriMeshEdgeCollapseOperation::before_check(const Tuple& t, TriMesh& m)
{
    return m.collapse_edge_before(t);
}
bool TriMeshEdgeCollapseOperation::after_check(const ExecuteReturnData& ret_data, TriMesh& m)
{
    return m.collapse_edge_after(ret_data.tuple);
}

std::string TriMeshEdgeCollapseOperation::name() const
{
    return "edge_collapse";
}

auto TriMeshConsolidateOperation::execute(const Tuple& t, TriMesh& m) -> ExecuteReturnData
{
    auto v_cnt = 0;
    std::vector<size_t> map_v_ids(m.vert_capacity(), -1);
    for (auto i = 0; i < m.vert_capacity(); i++) {
        if (m.m_vertex_connectivity[i].m_is_removed) continue;
        map_v_ids[i] = v_cnt;
        v_cnt++;
    }
    auto t_cnt = 0;
    std::vector<size_t> map_t_ids(m.tri_capacity(), -1);
    for (auto i = 0; i < m.tri_capacity(); i++) {
        if (m.m_tri_connectivity[i].m_is_removed) continue;
        map_t_ids[i] = t_cnt;
        t_cnt++;
    }
    v_cnt = 0;
    for (auto i = 0; i < m.vert_capacity(); i++) {
        if (m.m_vertex_connectivity[i].m_is_removed) continue;
        if (v_cnt != i) {
            assert(v_cnt < i);
            m.m_vertex_connectivity[v_cnt] = m.m_vertex_connectivity[i];
            if (m.p_vertex_attrs) m.p_vertex_attrs->move(i, v_cnt);
        }
        for (size_t& t_id : m.m_vertex_connectivity[v_cnt].m_conn_tris) {
            t_id = map_t_ids[t_id];
        }
        v_cnt++;
    }
    t_cnt = 0;
    for (int i = 0; i < m.tri_capacity(); i++) {
        if (m.m_tri_connectivity[i].m_is_removed) continue;

        if (t_cnt != i) {
            assert(t_cnt < i);
            m.m_tri_connectivity[t_cnt] = m.m_tri_connectivity[i];
            m.m_tri_connectivity[t_cnt].hash = 0;
            if (m.p_face_attrs) m.p_face_attrs->move(i, t_cnt);

            for (auto j = 0; j < 3; j++) {
                if (m.p_edge_attrs) m.p_edge_attrs->move(i * 3 + j, t_cnt * 3 + j);
            }
        }
        for (size_t& v_id : m.m_tri_connectivity[t_cnt].m_indices) v_id = map_v_ids[v_id];
        t_cnt++;
    }

    set_vertex_size(v_cnt, m);
    set_tri_size(t_cnt, m);

    // Resize user class attributes
    if (m.p_vertex_attrs) m.p_vertex_attrs->resize(m.vert_capacity());
    if (m.p_edge_attrs) m.p_edge_attrs->resize(m.tri_capacity() * 3);
    if (m.p_face_attrs) m.p_face_attrs->resize(m.tri_capacity());

    assert(m.check_edge_manifold());
    assert(m.check_mesh_connectivity_validity());
    ExecuteReturnData ret;
    ret.success = true;
    return ret;

}
bool TriMeshConsolidateOperation::before_check(const Tuple& t, TriMesh& m)
{
    return true;
}
bool TriMeshConsolidateOperation::after_check(const ExecuteReturnData& ret_data, TriMesh& m)
{
    return true;
}
std::string TriMeshConsolidateOperation::name() const
{
    return "consolidate";
}


//
// Created by Yixin Hu on 12/7/21.
//

#include "TetMesh.h"

using namespace wmtk;

TetMesh::Tuple TetMesh::Tuple::init_from_edge(const TetMesh& m, int tid, int local_eid)
{
    int vid = m.m_tet_connectivity[tid][m_local_edges[local_eid][0]];
    int fid = m_map_edge2face[local_eid];
    return Tuple(vid, local_eid, fid, tid);
}

TetMesh::Tuple TetMesh::Tuple::init_from_vertex(const TetMesh& m, int vid)
{
    int tid = m.m_vertex_connectivity[vid].m_conn_tets[0];
    int j = m.m_tet_connectivity[tid].find(vid);
    int eid = m_map_vertex2edge[j];
    int fid = m_map_edge2face[eid];
    return Tuple(vid, eid, fid, tid);
}

TetMesh::Tuple TetMesh::Tuple::init_from_tet(const TetMesh& m, int tid){
    int vid = m.m_tet_connectivity[tid][0];
    int eid = m_map_vertex2edge[0];
    int fid = m_map_edge2face[eid];
    return Tuple(vid, eid, fid, tid);
}

bool TetMesh::Tuple::is_valid(const TetMesh& m) const
{
    if (m.m_vertex_connectivity[m_vid].m_is_removed ||
        m.m_tet_connectivity[m_tid].m_is_removed)
        return false;
    return true;
}

void TetMesh::Tuple::update_version_number(const TetMesh& m)
{
    assert(m_timestamp >= m.m_tet_connectivity[m_tid].timestamp);
    m_timestamp = m.m_tet_connectivity[m_tid].timestamp;
}

int TetMesh::Tuple::get_version_number() { return m_timestamp; }

bool TetMesh::Tuple::is_version_number_valid(const TetMesh& m) const
{
    if (m_timestamp != m.m_tet_connectivity[m_tid].timestamp) return false;
    return true;
}

void TetMesh::Tuple::print_info() { logger().trace("tuple: {} {} {} {}", m_vid, m_eid, m_fid, m_tid); }

size_t TetMesh::Tuple::vid() const { return m_vid; } // update eid and fid

size_t TetMesh::Tuple::eid(const TetMesh& m) const
{
    int v1_id = m.m_tet_connectivity[m_tid][m_local_edges[m_eid][0]];
    int v2_id = m.m_tet_connectivity[m_tid][m_local_edges[m_eid][1]];
    auto n12_t_ids = set_intersection(m.m_vertex_connectivity[v1_id].m_conn_tets,
                                      m.m_vertex_connectivity[v2_id].m_conn_tets);
    int tid = *std::min_element(n12_t_ids.begin(), n12_t_ids.end());
    for(int j=0;j<6;j++){
        int tmp_v1_id = m.m_tet_connectivity[tid][m_local_edges[j][0]];
        int tmp_v2_id = m.m_tet_connectivity[tid][m_local_edges[j][1]];
        if(tmp_v1_id == v1_id && tmp_v2_id == v2_id
            ||tmp_v1_id == v2_id && tmp_v2_id == v1_id)
            return tid * 6 + j;
    }
    throw std::runtime_error("Tuple::eid() error");
}

size_t TetMesh::Tuple::fid(const TetMesh& m) const
{
    std::array<size_t, 3> v_ids = {
        {m.m_tet_connectivity[m_tid][m_local_faces[m_fid][0]],
         m.m_tet_connectivity[m_tid][m_local_faces[m_fid][1]],
         m.m_tet_connectivity[m_tid][m_local_faces[m_fid][2]]}};
    auto tmp = set_intersection(
        m.m_vertex_connectivity[v_ids[0]].m_conn_tets,
        m.m_vertex_connectivity[v_ids[1]].m_conn_tets);
    auto n12_t_ids = set_intersection(tmp, m.m_vertex_connectivity[v_ids[2]].m_conn_tets);
    if (n12_t_ids.size() == 1) {
        return m_tid * 4 + m_fid;
    }

    std::sort(v_ids.begin(), v_ids.end());
    int tid = *std::min_element(n12_t_ids.begin(), n12_t_ids.end());
    for (int j = 0; j < 4; j++) {
        std::array<size_t, 3> tmp_v_ids = {
            {m.m_tet_connectivity[tid][m_local_faces[j][0]],
             m.m_tet_connectivity[tid][m_local_faces[j][1]],
             m.m_tet_connectivity[tid][m_local_faces[j][2]]}};
        std::sort(tmp_v_ids.begin(), tmp_v_ids.end());
        if (tmp_v_ids == v_ids) return tid * 4 + j;
    }

    throw std::runtime_error("Tuple::fid() error");
}

size_t TetMesh::Tuple::tid() const { return m_tid; }

TetMesh::Tuple TetMesh::Tuple::switch_vertex(const TetMesh& m) const
{
    Tuple loc = *this;
    int l_vid1 = m_local_edges[m_eid][0];
    int l_vid2 = m_local_edges[m_eid][1];
    loc.m_vid = m.m_tet_connectivity[m_tid][l_vid1] == m_vid
                    ? m.m_tet_connectivity[m_tid][l_vid2]
                    : m.m_tet_connectivity[m_tid][l_vid1];

    return loc;
} // along edge

TetMesh::Tuple TetMesh::Tuple::switch_edge(const TetMesh& m) const
{
    Tuple loc = *this;
    for (int j = 0; j < 3; j++) {
        if (m_local_edges_in_a_face[m_fid][j] == m_eid) {
            loc.m_eid = m_local_edges_in_a_face[m_fid][(j + 1) % 3];
            return loc;
        }
    }
    assert("switch edge failed");
    return loc;
}

TetMesh::Tuple TetMesh::Tuple::switch_face(const TetMesh& m) const
{
    Tuple loc = *this;
    int l_v1_id = m_local_edges[m_eid][0];
    int l_v2_id = m_local_edges[m_eid][1];
    for (int j = 0; j < 4; j++) {
        if (j == m_fid) continue;
        int cnt = 0;
        for (int k = 0; k < 3; k++) {
            if (m_local_faces[j][k] == l_v1_id || m_local_faces[j][k] == l_v2_id) cnt++;
            if (cnt == 2) {
                loc.m_fid = j;
                return loc;
            }
        }
    }
    assert("switch face failed");
    return loc;
}

std::optional<TetMesh::Tuple> TetMesh::Tuple::switch_tetrahedron(const TetMesh& m) const
{
    // eid and fid are local, so they will be changed after switch tets
    int v1_id = m.m_tet_connectivity[m_tid][m_local_faces[m_fid][0]];
    int v2_id = m.m_tet_connectivity[m_tid][m_local_faces[m_fid][1]];
    int v3_id = m.m_tet_connectivity[m_tid][m_local_faces[m_fid][2]];
    auto tmp = set_intersection(
        m.m_vertex_connectivity[v1_id].m_conn_tets,
        m.m_vertex_connectivity[v2_id].m_conn_tets);
    auto n123_tids = set_intersection(tmp, m.m_vertex_connectivity[v3_id].m_conn_tets);
    if (n123_tids.size() == 1)
        return {};
    else {
        Tuple loc = *this;
        loc.m_tid = n123_tids[0] == m_tid ? n123_tids[1] : n123_tids[0];
        int j = m.m_tet_connectivity[loc.m_tid].find(loc.m_vid);
        loc.m_eid = m_map_vertex2edge[j];
        loc.m_fid = m_map_vertex2edge[loc.m_eid];
        return loc;
    }
}

std::vector<TetMesh::Tuple> TetMesh::Tuple::get_conn_tets(const TetMesh& m) const
{
        std::vector<Tuple> tets;
        //todo
        return tets;
}

std::array<TetMesh::Tuple, 4> TetMesh::Tuple::oriented_tet_vertices(const TetMesh& m) const
{
    std::array<Tuple, 4> vs;
    for (int j = 0; j < 4; j++) {
        vs[j].m_vid = m.m_tet_connectivity[m_tid][j];
        vs[j].m_eid = m_map_vertex2edge[j];
        vs[j].m_fid = m_map_edge2face[vs[j].m_eid];
        vs[j].m_tid = m_tid;
    }
    return vs;
}
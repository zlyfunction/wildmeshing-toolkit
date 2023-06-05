
#include <wmtk/operations/TriMeshEdgeSplitOperation.h>

namespace wmtk {
bool TriMeshEdgeSplitOperation::execute(TriMesh& m, const Tuple& t)
{
    auto& vertex_connectivity = this->vertex_connectivity(m);
    auto& tri_connectivity = this->tri_connectivity(m);


    //         v2                 v2
    //        /|\                /|\
    //       / | \              / | \
    //      /  |  \            /  |  \
    //     /   |   \          /   |   \
    //    /    |    \  ==>   /____nv___\
    //    \    |    /        \    |    /
    //     \   |   /          \   |   /
    //      \  |  /            \  |  /
    //       \ | /              \ | /
    //        \|/                \|/
    //         v1                 v1

    // First do the right side (which is guarnateed to exist) then do the left side
    //         v2                      v2
    //         |\                      |\
    //         | \                     | \
    //         |  \                    |  \
    //         |   \                   |nf1\
    //         |    \ fid1_v3   ==>    nv___\ fid1_v3
    //         |    /                  |    /
    //         |   /                   |f1 /
    //         |  /                    |  /
    //         | /                     | /
    //         |/                      |/
    //         v1                      v1

    // get local eid for return tuple construction
    auto eid = t.local_eid(m);
    // get the vids
    size_t vid1 = t.vid(m);
    size_t vid2 = m.switch_vertex(t).vid(m);
    size_t fid1 = t.fid(m);
    size_t fid1_vid3 = ((t.switch_vertex(m)).switch_edge(m)).switch_vertex(m).vid(m);

    // get an opt for if an opposing face exists
    std::optional<Tuple> fid2_opt = t.switch_face(m);


    for (auto vid : tri_connectivity[fid1].m_indices) {
        if ((vid != vid1) && (vid != vid2)) {
            assert(fid1_vid3 == vid);
            break;
        }
    }


    // create new face and vertex that will be used
    size_t new_vid = get_next_empty_slot_v(m);
    vertex_connectivity[new_vid].m_is_removed = false;
    size_t new_fid1 = get_next_empty_slot_t(m);
    tri_connectivity[new_fid1].m_is_removed = false;


    // shrink old triangle by substituting v2 with nv
    // requires:
    //  removing f1 from v2
    //  adding f1 to nv
    //  replacing v2 with nv in f1
    //
    //         v2
    //         |\
    //         | \
    //         |  \
    //         |   \
    //         |    \ fid1_v3   ==>    nv___  fid1_v3
    //         |    /                  |    /
    //         |   /                   |f1 /
    //         |  /                    |  /
    //         | /                     | /
    //         |/                      |/
    //         v1                      v1
    // (replace v2 with nv)
    // first work on the vids
    // the old triangles are connected to the vertex of t

    //  removing f1 from v2
    vector_erase(vertex_connectivity[vid2].m_conn_tris, fid1);
    //  adding f1 to nv
    vertex_connectivity[new_vid].m_conn_tris.push_back(fid1);
    //  replacing v2 with nv in f1
    size_t j = tri_connectivity[fid1].find(vid2);
    tri_connectivity[fid1].m_indices[j] = new_vid;


    // connect up the new face with the right ordering
    // attach nf1 to v2, nv, fid1_v3
    // construct nf1 with the right ordering
    //
    //            v2
    //            |\
    //            | \
    //            |  \
    //            |nf1\
    //            nv___\ fid1_v3

    // attach nf1 to v2, nv, fid1_v3
    for (size_t vid : {vid2, new_vid, fid1_vid3}) {
        vertex_connectivity[vid].m_conn_tris.push_back(new_fid1);
    }

    // construct nf1 with the right ordering
    // preserving ordering requires comparing with the original triangle
    // we use i,j,k to alias the proper indices.
    // j was evaluated on the original triangle, i,k are sought here
    //
    //       v2______ fid1_v3   ==>    j____ k
    //         |    /                  |    /
    //         |   /                   |f1 /
    //         |  /                    |  /
    //         | /                     | /
    //         |/                      |/
    //         v1                      i
    //
    //                                 ||
    //                                 || //orientation preserving
    //                                 \/
    //
    //          v2                     j
    //          |\                     |\
    //          | \                    | \
    //          |  \         <==       |  \
    //          |nf1\                  |nf1\
    //          nv___\ fid1_v3         i____\ k

    // identify fid1_v3 and v1 indices
    size_t i = tri_connectivity[fid1].find(vid1);
    size_t k = tri_connectivity[fid1].find(fid1_vid3);
    // replace values according to the above diagram
    tri_connectivity[new_fid1].m_indices[i] = new_vid;
    tri_connectivity[new_fid1].m_indices[j] = vid2;
    tri_connectivity[new_fid1].m_indices[k] = fid1_vid3;


    // To return a caanonical face for the returned tuple pick the min face
    size_t new_fid = std::min(fid1, new_fid1);

    // t.print_info();
    // if f2 exists then we have to do an analogous process to the above
    if (fid2_opt.has_value()) {
        const size_t fid2 = fid2_opt.value().fid(m);
        // removing f2 from v2
        // adding f2 to nv
        // replacing v2 with nv in f2
        // attach nf2 to v2, nv, fid2_v3
        // construct nf2 with the right ordering
        //         v2                    v2
        //        /|                     /|
        //       / |                    / |
        //      /  |                   /  |
        //     /   |         fid2_v3  /   |
        //    / f2 |       ==>       /____nv
        //    \    |                 \    |
        //     \   |                  \   |
        //      \  |                   \  |
        //       \ |                    \ |
        //        \|                     \|
        //         v1                     v1
        // get the id of the face
        // get the other vertex
        size_t fid2_vid3;
        for (auto vid : tri_connectivity[fid2].m_indices) {
            if ((vid != vid1) && (vid != vid2)) {
                fid2_vid3 = vid;
                break;
            }
        }

        size_t new_fid2 = get_next_empty_slot_t(m);
        tri_connectivity[new_fid2].m_is_removed = false;


        // removing f2 from v2
        vector_erase(vertex_connectivity[vid2].m_conn_tris, fid2);
        // adding f2 to nv
        vertex_connectivity[new_vid].m_conn_tris.push_back(fid2);
        // replacing v2 with nv in f2
        j = tri_connectivity[fid2].find(vid2);
        tri_connectivity[fid2].m_indices[j] = new_vid;
        // attach nf2 to v2, nv, fid2_v3
        for (const size_t vid : {vid2, new_vid, fid2_vid3}) {
            vertex_connectivity[vid].m_conn_tris.push_back(new_fid2);
        }

        // {vid1,vid2,fid2_vid3} => {i,j,k} => {nv, vid2, fid2_v3}
        // construct nf2 with the right ordering
        {
            const auto& tri_fid2 = tri_connectivity[fid2];
            i = tri_fid2.find(vid1);
            k = tri_fid2.find(fid2_vid3);
        }
        {
            auto& tri_fid2 = tri_connectivity[new_fid2];
            tri_fid2.m_indices[i] = new_vid;
            tri_fid2.m_indices[j] = vid2;
            tri_fid2.m_indices[k] = fid2_vid3;
        }

        {
            auto& tris = vertex_connectivity[fid2_vid3].m_conn_tris;
            std::sort(tris.begin(), tris.end());
        }


        // as part of returning a caanonical face for the return tuple we check with new fid
        new_fid = std::min(new_fid, new_fid2);

        // update all modified hashes
        tri_connectivity[fid2].hash++;
        tri_connectivity[new_fid2].hash++;
    }
    // clean up new vertex connectivity orders for vertices modified
    for (size_t vid : {vid2, new_vid, fid1_vid3}) {
        std::sort(
            vertex_connectivity[vid].m_conn_tris.begin(),
            vertex_connectivity[vid].m_conn_tris.end());
    }
    // update all modified hashes
    tri_connectivity[fid1].hash++;
    tri_connectivity[new_fid1].hash++;

    // find the position of the new vertex in the selected face
    int l = tri_connectivity[new_fid].find(new_vid);
    const Tuple return_tuple = Tuple(vid1, eid, fid1, m);
    assert(return_tuple.is_valid(m));

#if !defined(NDEBUG)
    auto new_vertex = Tuple(new_vid, (l + 2) % 3, new_fid, m);
    assert(new_vertex.is_valid(m));
    // assert(new_vertex == this->new_vertex(m));
#endif

    set_return_tuple(return_tuple);
    return true;
}

bool TriMeshEdgeSplitOperation::before(TriMesh& m, const Tuple& t)
{
    return true;
}
bool TriMeshEdgeSplitOperation::after(TriMesh& m)
{
    return true;
}
std::string TriMeshEdgeSplitOperation::name() const
{
    return "edge_split";
}

auto TriMeshEdgeSplitOperation::new_vertex(const TriMesh& m) const -> Tuple
{
    assert(bool(*this));
    const std::optional<Tuple>& new_tup = get_return_tuple_opt();
    assert(new_tup.has_value());
    return new_tup.value().switch_vertex(m);
}

auto TriMeshEdgeSplitOperation::original_endpoints(TriMesh& m, const Tuple& t) const
    -> std::array<Tuple, 2>
{
    // diamond below encodes vertices with lower alpha
    // edges with num
    // faces with upper alpha
    // (only encodes simplices adjacent to vertex c
    //   a
    //  A1B
    // b2c3d
    //  C4D
    //   e
    //
    // initially e4D
    // switch_vertex -> c4D
    // switch_edge -> c3D
    // switch_face -> c3B
    // switch_edge -> c1B
    // switch_vertex -> a1B
    auto face_opt = t.switch_vertex(m).switch_edge(m).switch_face(m);

    assert(face_opt);

    return {{t, face_opt->switch_edge(m).switch_vertex(m)}};
}
auto TriMeshEdgeSplitOperation::modified_triangles(const TriMesh& m) const -> std::vector<Tuple>
{
    if (!bool(*this)) {
        return {};
    }

    const Tuple new_v = new_vertex(m);

    return m.get_one_ring_tris_for_vertex(new_v);
}
} // namespace wmtk

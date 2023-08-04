#pragma once
#include <wmtk/utils/Logger.hpp>
#include "SimplicialComplex.hpp"
#include "TriMesh.hpp"
#include "Tuple.hpp"
namespace wmtk {
class TriMesh::TriMeshOperationExecutor
{
public:
    TriMeshOperationExecutor(TriMesh& m, const Tuple& operating_tuple);
    void delete_simplices();
    void update_cell_hash();

    std::array<Accessor<char>, 3> flag_accessors;
    Accessor<long> ff_accessor;
    Accessor<long> fe_accessor;
    Accessor<long> fv_accessor;
    Accessor<long> vf_accessor;
    Accessor<long> ef_accessor;
    Accessor<long> hash_accessor;


    //           C
    //         /  \ .
    //    F1  /    \  F2
    //       /      \ .
    //      /        \ .
    //     A----------B
    //      \        /
    //       \      /
    //    F1' \    / F2'
    //         \  /
    //          C'
    // the neighbors are stored in the order of A, B, C, D if they exist
    // vid, ear fid (-1 if it doesn't exit), ear eid

    /**
     * An ear is a face that is adjacent to a face that is incident to the edge on which the
     * operation is performed. In other words, the ears are the neighboring faces to the ones that
     * will be deleted by the operation.
     */
    struct EarFace
    {
        long fid = -1; // global fid of the ear, -1 if it doesn't exist
        long eid = -1; // global eid of the ear, -1 if it doesn't exist
    };

    /**
     * Data on the incident face relevant for performing operations.
     */
    struct IncidentFaceData
    {
        long opposite_vid = -1; // opposing vid
        long fid = -1; // the face that will be deleted
        long f0 = -1;
        long f1 = -1;
        std::array<EarFace, 2> ears; // ear
    };

    /**
     * @brief gather all simplices that are deleted in a split
     *
     * The deleted simplices are exactly the open star of the edge
     */
    static const SimplicialComplex get_split_simplices_to_delete(
        const Tuple& tuple,
        const TriMesh& m);

    /**
     * @brief gather all simplices that are deleted in a collapse
     *
     * The deleted simplices are the intersection of the open star of the vertex and the closed star
     * of the edge. This comes down to one vertex, three edges, and two faces if the edge is on the
     * interior. On the boundary it is one vertex, two edges, and one face.
     */
    static const SimplicialComplex get_collapse_simplices_to_delete(
        const Tuple& tuple,
        const TriMesh& m);

    std::vector<IncidentFaceData>& incident_face_datas() { return m_incident_face_datas; }

    const std::array<long, 2>& incident_vids() const { return m_spine_vids; }

    const long operating_edge_id() const { return m_operating_edge_id; }

    void update_fid_in_ear(
        const long ear_fid,
        const long new_face_fid,
        const long old_fid,
        const long eid);

    void merge(const long& new_vid);
    Tuple split_edge();

    /**
     * @brief
     *
     * @param new_fids_top the two new fids on the top sides of AB generated by split_edge(AB)
     * @param new_fids_bottom the two new fids on the bottom sides of AB generated by split_edge(AB)
     */
    // return the two new fids in order
    std::array<long, 2> glue_new_triangle_topology(
        const long new_vid,
        const std::vector<long>& spine_eids,
        IncidentFaceData& face_data);
    void glue_new_faces_across_AB(
        const std::array<long, 2> new_fids_top,
        const std::array<long, 2> new_fids_bottom);
    std::vector<long> request_simplex_indices(const PrimitiveType type, long count);

    SimplicialComplex simplices_to_delete;
    std::vector<long> cell_ids_to_update_hash;
    TriMesh& m_mesh;
    Tuple m_operating_tuple;

private:
    // common simplicies
    std::array<long, 2> m_spine_vids; // V_A_id, V_B_id;
    long m_operating_edge_id;

    // simplices required per-face
    std::vector<IncidentFaceData> m_incident_face_datas;

    IncidentFaceData get_incident_face_data(Tuple t);
};
} // namespace wmtk

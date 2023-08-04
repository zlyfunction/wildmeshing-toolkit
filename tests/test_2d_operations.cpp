#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <wmtk/Accessor.hpp>
#include <wmtk/TriMeshOperationExecutor.hpp>
#include <wmtk/utils/Logger.hpp>
#include "tools/DEBUG_TriMesh.hpp"
#include "tools/TriMesh_examples.hpp"


using namespace wmtk;
using namespace wmtk::tests;

using TM = TriMesh;
using MapResult = typename Eigen::Matrix<long, Eigen::Dynamic, 1>::MapType;
using TMOE = decltype(std::declval<DEBUG_TriMesh>().get_tmoe(wmtk::Tuple()));

constexpr PrimitiveType PV = PrimitiveType::Vertex;
constexpr PrimitiveType PE = PrimitiveType::Edge;
constexpr PrimitiveType PF = PrimitiveType::Face;

TEST_CASE("incident_face_data", "[operations][2D]")
{
    SECTION("single_face")
    {
        //         0
        //        / \   .
        //       2   1  \ .
        //      /  0  \  \|
        //     /       \ .
        //  1  ----0---- 2
        //
        DEBUG_TriMesh m = single_triangle();
        REQUIRE(m.is_connectivity_valid());

        const Tuple edge = m.edge_tuple_between_v1_v2(0, 2, 0);
        REQUIRE(m._debug_id(edge, PV) == 0);
        REQUIRE(m._debug_id(edge, PF) == 0);
        REQUIRE(m._debug_id(m.switch_tuple(edge, PV), PV) == 2);
        auto executor = m.get_tmoe(edge);

        const std::vector<TMOE::IncidentFaceData>& face_datas = executor.incident_face_datas();
        REQUIRE(face_datas.size() == 1);
        const TMOE::IncidentFaceData& face_data = face_datas[0];
        CHECK(face_data.opposite_vid == 1);
        CHECK(face_data.fid == 0);
        REQUIRE(face_data.ears.size() == 2);
        TMOE::EarFace ear1 = face_data.ears[0];
        TMOE::EarFace ear2 = face_data.ears[1];
        CHECK(ear1.fid == -1);
        CHECK(ear1.eid > -1);
        CHECK(ear2.fid == -1);
        CHECK(ear2.eid > -1);
    }
    SECTION("one_ear")
    {
        //  3--1--- 0
        //   |     / \ .
        //   2 f1 /2   1
        //   |  0/ f0  \ .
        //   |  /       \ .
        //  1  ----0---- 2
        //
        DEBUG_TriMesh m = one_ear();

        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);
        const std::vector<TMOE::IncidentFaceData>& face_datas = executor.incident_face_datas();
        REQUIRE(face_datas.size() == 1);
        const TMOE::IncidentFaceData& face_data = face_datas[0];
        CHECK(face_data.opposite_vid == 0);
        CHECK(face_data.fid == 0);
        REQUIRE(face_data.ears.size() == 2);
        TMOE::EarFace ear1 = face_data.ears[0];
        TMOE::EarFace ear2 = face_data.ears[1];
        CHECK(ear1.fid == 1);
        CHECK(ear1.eid > -1);
        CHECK(ear2.fid == -1);
        CHECK(ear2.eid > -1);
    }
    SECTION("two_ears")
    {
        //  3--1--- 0 --1- 4
        //   |     / \     |
        //   2 f1 /2 1\ f2 |
        //   |  0/ f0  \1  0
        //   |  /       \  |
        //   1  ----0----  2
        //
        DEBUG_TriMesh m = two_neighbors();

        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);
        const std::vector<TMOE::IncidentFaceData>& face_datas = executor.incident_face_datas();
        REQUIRE(face_datas.size() == 1);
        const TMOE::IncidentFaceData& face_data = face_datas[0];
        CHECK(face_data.opposite_vid == 0);
        CHECK(face_data.fid == 0);
        REQUIRE(face_data.ears.size() == 2);
        TMOE::EarFace ear1 = face_data.ears[0];
        TMOE::EarFace ear2 = face_data.ears[1];
        CHECK(ear1.fid == 1);
        CHECK(ear1.eid > -1);
        CHECK(ear2.fid == 2);
        CHECK(ear2.eid > -1);
    }
}

TEST_CASE("get_split_simplices_to_delete", "[operations][split][2D]")
{
    SECTION("single_triangle")
    {
        const DEBUG_TriMesh m = single_triangle();
        const Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);

        SimplicialComplex sc_to_delete = TMOE::get_split_simplices_to_delete(edge, m);
        const auto& simplices = sc_to_delete.get_simplices();

        REQUIRE(simplices.size() == 2);
        REQUIRE(sc_to_delete.get_vertices().size() == 0);
        REQUIRE(sc_to_delete.get_edges().size() == 1);
        REQUIRE(sc_to_delete.get_faces().size() == 1);

        const Simplex edge_to_delete = *sc_to_delete.get_edges().begin();
        CHECK(m._debug_id(edge_to_delete) == m._debug_id(edge, PE));
        const Simplex face_to_delete = *sc_to_delete.get_faces().begin();
        CHECK(m._debug_id(face_to_delete) == m._debug_id(edge, PF));
    }
    SECTION("hex_plus_two")
    {
        const DEBUG_TriMesh m = hex_plus_two();
        const Tuple edge = m.edge_tuple_between_v1_v2(4, 5, 2);

        SimplicialComplex sc_to_delete = TMOE::get_split_simplices_to_delete(edge, m);
        const auto& simplices = sc_to_delete.get_simplices();

        REQUIRE(simplices.size() == 3);
        REQUIRE(sc_to_delete.get_vertices().size() == 0);
        REQUIRE(sc_to_delete.get_edges().size() == 1);
        REQUIRE(sc_to_delete.get_faces().size() == 2);

        const Simplex edge_to_delete = *sc_to_delete.get_edges().begin();
        CHECK(m._debug_id(edge_to_delete) == m._debug_id(edge, PE));

        // compare expected face ids with the actual ones that should be deleted
        std::set<long> fid_expected;
        fid_expected.insert(m._debug_id(edge, PF));
        fid_expected.insert(m._debug_id(m.switch_face(edge), PF));

        std::set<long> fid_actual;
        for (const Simplex& f : sc_to_delete.get_faces()) {
            const long fid = m._debug_id(f);
            CHECK(fid_expected.find(fid) != fid_expected.end());
            fid_actual.insert(fid);
        }
        CHECK(fid_actual.size() == fid_expected.size());
    }
}

TEST_CASE("get_collapse_simplices_to_delete", "[operations][collapse][2D]")
{
    SECTION("interior_edge")
    {
        const DEBUG_TriMesh m = edge_region();
        Tuple edge = m.edge_tuple_between_v1_v2(4, 5, 2);

        SimplicialComplex sc_to_delete = TMOE::get_collapse_simplices_to_delete(edge, m);
        const auto& simplices = sc_to_delete.get_simplices();

        REQUIRE(simplices.size() == 6);
        REQUIRE(sc_to_delete.get_vertices().size() == 1);
        REQUIRE(sc_to_delete.get_edges().size() == 3);
        REQUIRE(sc_to_delete.get_faces().size() == 2);

        // V
        const Simplex vertex_to_delete = *sc_to_delete.get_vertices().begin();
        CHECK(m._debug_id(vertex_to_delete) == m._debug_id(edge, PV));

        // E
        std::set<long> eid_expected;
        eid_expected.insert(m._debug_id(edge, PE));
        eid_expected.insert(m._debug_id(m.switch_edge(edge), PE));
        eid_expected.insert(m._debug_id(m.switch_edge(m.switch_face(edge)), PE));

        std::set<long> eid_actual;
        for (const Simplex& e : sc_to_delete.get_edges()) {
            const long eid = m._debug_id(e);
            CHECK(eid_expected.find(eid) != eid_expected.end());
            eid_actual.insert(eid);
        }
        CHECK(eid_actual.size() == eid_expected.size());

        // F
        std::set<long> fid_expected;
        fid_expected.insert(m._debug_id(edge, PF));
        fid_expected.insert(m._debug_id(m.switch_face(edge), PF));

        std::set<long> fid_actual;
        for (const Simplex& f : sc_to_delete.get_faces()) {
            const long fid = m._debug_id(f);
            CHECK(fid_expected.find(fid) != fid_expected.end());
            fid_actual.insert(fid);
        }
        CHECK(fid_actual.size() == fid_expected.size());
    }
    SECTION("boundary_edge")
    {
        const DEBUG_TriMesh m = edge_region();
        Tuple edge = m.edge_tuple_between_v1_v2(7, 8, 6);

        SimplicialComplex sc_to_delete = TMOE::get_collapse_simplices_to_delete(edge, m);
        const auto& simplices = sc_to_delete.get_simplices();

        REQUIRE(simplices.size() == 4);
        REQUIRE(sc_to_delete.get_vertices().size() == 1);
        REQUIRE(sc_to_delete.get_edges().size() == 2);
        REQUIRE(sc_to_delete.get_faces().size() == 1);

        // V
        const Simplex vertex_to_delete = *sc_to_delete.get_vertices().begin();
        CHECK(m._debug_id(vertex_to_delete) == m._debug_id(edge, PV));

        // E
        std::set<long> eid_expected;
        eid_expected.insert(m._debug_id(edge, PE));
        eid_expected.insert(m._debug_id(m.switch_edge(edge), PE));

        std::set<long> eid_actual;
        for (const Simplex& e : sc_to_delete.get_edges()) {
            const long eid = m._debug_id(e);
            CHECK(eid_expected.find(eid) != eid_expected.end());
            eid_actual.insert(eid);
        }
        CHECK(eid_actual.size() == eid_expected.size());

        // F
        const Simplex face_to_delete = *sc_to_delete.get_faces().begin();
        CHECK(m._debug_id(face_to_delete) == m._debug_id(edge, PF));
    }
    SECTION("interior_edge_incident_to_boundary")
    {
        const DEBUG_TriMesh m = edge_region();
        Tuple edge = m.edge_tuple_between_v1_v2(7, 4, 5);

        SimplicialComplex sc_to_delete = TMOE::get_collapse_simplices_to_delete(edge, m);
        const auto& simplices = sc_to_delete.get_simplices();

        REQUIRE(simplices.size() == 6);
        REQUIRE(sc_to_delete.get_vertices().size() == 1);
        REQUIRE(sc_to_delete.get_edges().size() == 3);
        REQUIRE(sc_to_delete.get_faces().size() == 2);

        // V
        const Simplex vertex_to_delete = *sc_to_delete.get_vertices().begin();
        CHECK(m._debug_id(vertex_to_delete) == m._debug_id(edge, PV));

        // E
        std::set<long> eid_expected;
        eid_expected.insert(m._debug_id(edge, PE));
        eid_expected.insert(m._debug_id(m.switch_edge(edge), PE));
        eid_expected.insert(m._debug_id(m.switch_edge(m.switch_face(edge)), PE));

        std::set<long> eid_actual;
        for (const Simplex& e : sc_to_delete.get_edges()) {
            const long eid = m._debug_id(e);
            CHECK(eid_expected.find(eid) != eid_expected.end());
            eid_actual.insert(eid);
        }
        CHECK(eid_actual.size() == eid_expected.size());

        // F
        std::set<long> fid_expected;
        fid_expected.insert(m._debug_id(edge, PF));
        fid_expected.insert(m._debug_id(m.switch_face(edge), PF));

        std::set<long> fid_actual;
        for (const Simplex& f : sc_to_delete.get_faces()) {
            const long fid = m._debug_id(f);
            CHECK(fid_expected.find(fid) != fid_expected.end());
            fid_actual.insert(fid);
        }
        CHECK(fid_actual.size() == fid_expected.size());
    }
}

TEST_CASE("delete_simplices", "[operations][2D]")
{
    // delete for split

    // things can be marked as deleted but will still have the connectivity information
    DEBUG_TriMesh m = two_neighbors();
    REQUIRE(m.is_connectivity_valid());
    Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
    std::vector<std::vector<long>> simplices_to_delete(3);
    simplices_to_delete[1].emplace_back(m._debug_id(edge, PE));
    simplices_to_delete[2].emplace_back(m._debug_id(edge, PF));

    auto executor = m.get_tmoe(edge);

    // new way of getting simplices
    executor.simplices_to_delete = TMOE::get_split_simplices_to_delete(edge, m);

    executor.delete_simplices();
    REQUIRE(executor.flag_accessors[1].scalar_attribute(edge) == 0);
    REQUIRE(executor.flag_accessors[2].scalar_attribute(edge) == 0);
    REQUIRE(executor.ff_accessor.vector_attribute(edge)[0] == -1);
    REQUIRE(executor.ff_accessor.vector_attribute(edge)[1] == 2);
    REQUIRE(executor.ff_accessor.vector_attribute(edge)[2] == 1);
    REQUIRE(executor.ef_accessor.scalar_attribute(edge) == 0);
}

TEST_CASE("operation_state", "[operations][2D]")
{
    SECTION("single_face")
    {
        DEBUG_TriMesh m = single_triangle();

        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(0, 2, 0);
        REQUIRE(m._debug_id(edge, PV) == 0);
        REQUIRE(m._debug_id(edge, PF) == 0);
        REQUIRE(m._debug_id(m.switch_tuple(edge, PV), PV) == 2);
        auto executor = m.get_tmoe(edge);

        REQUIRE(executor.flag_accessors.size() == 3);
        REQUIRE(executor.incident_vids().size() == 2);
        REQUIRE(executor.incident_vids()[0] == 0);
        REQUIRE(executor.incident_vids()[1] == 2);
        REQUIRE(executor.operating_edge_id() == m._debug_id(edge, PE));
        REQUIRE(executor.incident_face_datas().size() == 1);
    }
    SECTION("one_ear")
    {
        DEBUG_TriMesh m = one_ear();

        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        REQUIRE(executor.flag_accessors.size() == 3);
        REQUIRE(executor.incident_vids().size() == 2);
        REQUIRE(executor.incident_vids()[0] == 1);
        REQUIRE(executor.incident_vids()[1] == 2);
        REQUIRE(executor.operating_edge_id() == m._debug_id(edge, PE));
        REQUIRE(executor.incident_face_datas().size() == 1);

        REQUIRE(executor.incident_face_datas()[0].ears.size() == 2);
    }
    SECTION("interior_edge")
    {
        //  3--1--- 0
        //   |     / \ .
        //   2 f1 /2   1
        //   |  0/ f0  \ .
        //   |  /       \ .
        //  1  ----0---- 2
        //     \        /
        //      \  f2  /
        //       \    /
        //        \  /
        //         4
        DEBUG_TriMesh m = interior_edge();

        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        REQUIRE(executor.flag_accessors.size() == 3);
        REQUIRE(executor.incident_vids().size() == 2);
        REQUIRE(executor.incident_vids()[0] == 1);
        REQUIRE(executor.incident_vids()[1] == 2);
        REQUIRE(executor.operating_edge_id() == m._debug_id(edge, PE));
        REQUIRE(executor.incident_face_datas().size() == 2);

        REQUIRE(executor.incident_face_datas()[0].opposite_vid == 0);
        REQUIRE(executor.incident_face_datas()[0].fid == 0);
        REQUIRE(executor.incident_face_datas()[0].ears.size() == 2);
        TMOE::EarFace ear1 = executor.incident_face_datas()[0].ears[0];
        TMOE::EarFace ear2 = executor.incident_face_datas()[0].ears[1];
        REQUIRE(ear1.fid == 1);
        REQUIRE(ear1.eid > -1);
        REQUIRE(ear2.fid == -1);
        REQUIRE(ear2.eid > -1);

        REQUIRE(executor.incident_face_datas()[1].opposite_vid == 4);
        REQUIRE(executor.incident_face_datas()[1].fid == 2);
        REQUIRE(executor.incident_face_datas()[1].ears.size() == 2);
        ear1 = executor.incident_face_datas()[1].ears[0];
        ear2 = executor.incident_face_datas()[1].ears[1];
        REQUIRE(ear1.fid == -1);
        REQUIRE(ear1.eid > -1);
        REQUIRE(ear2.fid == -1);
        REQUIRE(ear2.eid > -1);
    }
}
TEST_CASE("glue_ear_to_face", "[operations][2D]")
{
    //    0---1---2
    //   / \ / \ / \ .
    //  3---4---5---6
    //   \ / \ /  .
    //    7---8
    DEBUG_TriMesh m = hex_plus_two();

    REQUIRE(m.is_connectivity_valid());
    const Tuple edge = m.edge_tuple_between_v1_v2(4, 5, 2);
    const Tuple left_ear_edge = m.switch_tuple(edge, PE);
    REQUIRE(m._debug_id(left_ear_edge, PV) == 4);
    REQUIRE(m._debug_id(m.switch_tuple(left_ear_edge, PV), PV) == 1);
    auto executor = m.get_tmoe(edge);
    auto ff_accessor_before = m.create_base_accessor<long>(m.f_handle(PF));
    REQUIRE(ff_accessor_before.vector_attribute(1)(2) == 2);
    executor.update_fid_in_ear(1, 3, 2, m._debug_id(edge, PE));
    auto ff_accessor_after = m.create_base_accessor<long>(m.f_handle(PF));
    REQUIRE(ff_accessor_after.vector_attribute(1)(2) == 3);
}
TEST_CASE("hash_update", "[operations][2D][.]")
{
    REQUIRE(false);
}

//////////// SPLIT TESTS ////////////
TEST_CASE("glue_new_faces_across_AB", "[operations][2D]")
{
    // test the assumption of correct orientation
    // new face correspondance accross AB
    SECTION("single_face")
    {
        // when the edge is on the boundary (indcated by FaceDatas size), there is no glue
        // across AB
        DEBUG_TriMesh m = single_triangle();
        REQUIRE(m.is_connectivity_valid());
        const Tuple edge = m.edge_tuple_between_v1_v2(0, 2, 0);
        REQUIRE(m._debug_id(edge, PV) == 0);
        REQUIRE(m._debug_id(edge, PF) == 0);
        REQUIRE(m._debug_id(m.switch_tuple(edge, PV), PV) == 2);
        auto executor = m.get_tmoe(edge);
        REQUIRE(executor.incident_face_datas().size() == 1);
    }
    SECTION("interior_edge")
    {
        DEBUG_TriMesh m = interior_edge();
        m.reserve_attributes(PF, 10);
        REQUIRE(m.is_connectivity_valid());
        const Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        REQUIRE(executor.incident_face_datas().size() == 2);

        const auto new_fids = executor.request_simplex_indices(PF, 4);
        const std::array<long, 2> new_fids_top = {new_fids[0], new_fids[1]};
        const std::array<long, 2> new_fids_bottom = {new_fids[2], new_fids[3]};
        executor.glue_new_faces_across_AB(new_fids_top, new_fids_bottom);


        long local_eid_top = 0;
        long local_eid_bottom = 1;

        auto ff_accessor = m.create_base_accessor<long>(m.f_handle(PF));

        REQUIRE(ff_accessor.vector_attribute(new_fids_top[0])[local_eid_top] == new_fids_bottom[0]);

        REQUIRE(ff_accessor.vector_attribute(new_fids_top[1])[local_eid_top] == new_fids_bottom[1]);

        REQUIRE(
            ff_accessor.vector_attribute(new_fids_bottom[0])[local_eid_bottom] == new_fids_top[0]);

        REQUIRE(
            ff_accessor.vector_attribute(new_fids_bottom[1])[local_eid_bottom] == new_fids_top[1]);
    }
}

TEST_CASE("glue_new_triangle", "[operations][2D]")
{
    SECTION("boundary_edge")
    {
        DEBUG_TriMesh m = single_triangle();
        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        //  create new vertex
        std::vector<long> new_vids = executor.request_simplex_indices(PV, 1);
        REQUIRE(new_vids.size() == 1);
        const long v_new = new_vids[0];

        // create new edges
        std::vector<long> spine_eids = executor.request_simplex_indices(PE, 2);
        REQUIRE(spine_eids.size() == 2);

        std::vector<std::array<long, 2>> new_fids;
        REQUIRE(executor.incident_face_datas().size() == 1);
        for (size_t i = 0; i < executor.incident_face_datas().size(); ++i) {
            auto& face_data = executor.incident_face_datas()[i];
            // glue the topology
            std::array<long, 2> new_fid_per_face =
                executor.glue_new_triangle_topology(v_new, spine_eids, face_data);
            new_fids.emplace_back(new_fid_per_face);
        }
        REQUIRE(new_fids.size() == 1);

        const long& f0 = new_fids[0][0];
        const long& f1 = new_fids[0][1];
        const long& se0 = spine_eids[0];
        const long& se1 = spine_eids[1];
        const long& ee0 = executor.incident_face_datas()[0].ears[0].eid;
        const long& ee1 = executor.incident_face_datas()[0].ears[1].eid;

        auto fv_accessor = m.create_base_accessor<long>(m.f_handle(PV));
        const auto fv0 = fv_accessor.vector_attribute(f0);
        const auto fv1 = fv_accessor.vector_attribute(f1);
        CHECK(fv0[0] == 0);
        CHECK(fv0[1] == 1);
        CHECK(fv0[2] == v_new);

        CHECK(fv1[0] == 0);
        CHECK(fv1[1] == v_new);
        CHECK(fv1[2] == 2);

        // the new fids generated are in top-down left-right order
        auto ff_accessor = m.create_base_accessor<long>(m.f_handle(PF));
        const auto ff0 = ff_accessor.vector_attribute(f0);
        const auto ff1 = ff_accessor.vector_attribute(f1);
        CHECK(ff0[0] == -1);
        CHECK(ff0[1] == f1);
        CHECK(ff0[2] == -1);

        CHECK(ff1[0] == -1);
        CHECK(ff1[1] == -1);
        CHECK(ff1[2] == f0);

        auto fe_accessor = m.create_base_accessor<long>(m.f_handle(PE));
        const auto fe0 = fe_accessor.vector_attribute(f0);
        const auto fe1 = fe_accessor.vector_attribute(f1);

        CHECK(fe0[0] == se0);
        CHECK(fe0[1] == 5);
        CHECK(fe0[2] == ee0);

        CHECK(fe1[0] == se1);
        CHECK(fe1[1] == ee1);
        CHECK(fe1[2] == 5);

        auto vf_accessor = m.create_base_accessor<long>(m.vf_handle());
        CHECK(vf_accessor.scalar_attribute(v_new) == f0);
        CHECK(vf_accessor.scalar_attribute(0) == f0);
        CHECK(vf_accessor.scalar_attribute(1) == f0);
        CHECK(vf_accessor.scalar_attribute(2) == f1);

        auto ef_accessor = m.create_base_accessor<long>(m.ef_handle());
        CHECK(ef_accessor.scalar_attribute(se0) == f0);
        CHECK(ef_accessor.scalar_attribute(se1) == f1);
        CHECK(ef_accessor.scalar_attribute(ee0) == f0);
        CHECK(ef_accessor.scalar_attribute(ee1) == f1);
        CHECK(ef_accessor.scalar_attribute(5) == f0);
    }
    SECTION("interior_edge")
    {
        // old faces are not recycled
        DEBUG_TriMesh m = interior_edge();
        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        // create new vertex
        std::vector<long> new_vids = executor.request_simplex_indices(PV, 1);
        REQUIRE(new_vids.size() == 1);
        const long v_new = new_vids[0];

        // create new edges
        std::vector<long> spine_eids = executor.request_simplex_indices(PE, 2);
        REQUIRE(spine_eids.size() == 2);

        std::vector<std::array<long, 2>> new_fids;
        for (size_t i = 0; i < executor.incident_face_datas().size(); ++i) {
            auto& face_data = executor.incident_face_datas()[i];
            // glue the topology
            std::array<long, 2> new_fid_per_face =
                executor.glue_new_triangle_topology(v_new, spine_eids, face_data);
            new_fids.emplace_back(new_fid_per_face);
        }
        REQUIRE(new_fids.size() == 2);

        auto fv_accessor = m.create_base_accessor<long>(m.f_handle(PV));
        auto fe_accessor = m.create_base_accessor<long>(m.f_handle(PE));
        auto ff_accessor = m.create_base_accessor<long>(m.f_handle(PF));

        auto vf_accessor = m.create_base_accessor<long>(m.vf_handle());

        auto ef_accessor = m.create_base_accessor<long>(m.ef_handle());

        const long& se0 = spine_eids[0];
        const long& se1 = spine_eids[1];

        // top
        {
            const long& f1 = new_fids[0][1];
            const long& f0 = new_fids[0][0];
            const long& ee0 = executor.incident_face_datas()[0].ears[0].eid;
            const long& ee1 = executor.incident_face_datas()[0].ears[1].eid;

            const auto fv0 = fv_accessor.vector_attribute(f0);
            const auto fv1 = fv_accessor.vector_attribute(f1);
            CHECK(fv0[0] == 0);
            CHECK(fv0[1] == 1);
            CHECK(fv0[2] == v_new);

            CHECK(fv1[0] == 0);
            CHECK(fv1[1] == v_new);
            CHECK(fv1[2] == 2);

            const auto ff0 = ff_accessor.vector_attribute(f0);
            const auto ff1 = ff_accessor.vector_attribute(f1);
            CHECK(ff0[0] == executor.incident_face_datas()[1].fid);
            CHECK(ff0[1] == f1);
            CHECK(ff0[2] == executor.incident_face_datas()[0].ears[0].fid);
            CHECK(ff1[0] == executor.incident_face_datas()[1].fid);
            CHECK(ff1[1] == -1);
            CHECK(ff1[2] == f0);

            const auto fe0 = fe_accessor.vector_attribute(f0);
            const auto fe1 = fe_accessor.vector_attribute(f1);
            CHECK(fe0[0] == spine_eids[0]);
            CHECK(fe0[1] == 9);
            CHECK(fe0[2] == ee0);

            CHECK(fe1[0] == spine_eids[1]);
            CHECK(fe1[1] == ee1);
            CHECK(fe1[2] == 9);

            CHECK(vf_accessor.scalar_attribute(0) == f0);

            CHECK(ef_accessor.scalar_attribute(ee0) == f0);
            CHECK(ef_accessor.scalar_attribute(ee1) == f1);
            CHECK(ef_accessor.scalar_attribute(9) == f0);
        }
        // bottom
        {
            const long& f0 = new_fids[1][0];
            const long& f1 = new_fids[1][1];
            const long& ee0 = executor.incident_face_datas()[1].ears[0].eid;
            const long& ee1 = executor.incident_face_datas()[1].ears[1].eid;

            const auto fv0 = fv_accessor.vector_attribute(f0);
            const auto fv1 = fv_accessor.vector_attribute(f1);

            CHECK(fv0[0] == 1);
            CHECK(fv0[1] == 4);
            CHECK(fv0[2] == v_new);

            CHECK(fv1[0] == v_new);
            CHECK(fv1[1] == 4);
            CHECK(fv1[2] == 2);

            const auto ff0 = ff_accessor.vector_attribute(f0);
            const auto ff1 = ff_accessor.vector_attribute(f1);
            CHECK(ff0[0] == f1);
            CHECK(ff0[1] == executor.incident_face_datas()[0].fid);
            CHECK(ff0[2] == -1);

            CHECK(ff1[0] == -1);
            CHECK(ff1[1] == executor.incident_face_datas()[0].fid);
            CHECK(ff1[2] == f0);

            const auto fe0 = fe_accessor.vector_attribute(f0);
            const auto fe1 = fe_accessor.vector_attribute(f1);
            CHECK(fe0[0] == 10);
            CHECK(fe0[1] == se0);
            CHECK(fe0[2] == ee0);

            CHECK(fe1[0] == ee1);
            CHECK(fe1[1] == se1);
            CHECK(fe1[2] == 10);

            CHECK(vf_accessor.scalar_attribute(v_new) == f0);
            CHECK(vf_accessor.scalar_attribute(4) == f0);
            CHECK(vf_accessor.scalar_attribute(1) == f0);
            CHECK(vf_accessor.scalar_attribute(2) == f1);

            CHECK(ef_accessor.scalar_attribute(se0) == f0);
            CHECK(ef_accessor.scalar_attribute(se1) == f1);
            CHECK(ef_accessor.scalar_attribute(ee0) == f0);
            CHECK(ef_accessor.scalar_attribute(ee1) == f1);
            CHECK(ef_accessor.scalar_attribute(10) == f0);
        }
    }
}

TEST_CASE("simplices_to_delete_for_split", "[operations][2D]")
{
    SECTION("boundary_edge")
    {
        // old faces are not recycled
        DEBUG_TriMesh m;
        {
            //         0
            //        / \ 
            //       /2   1
            //      / f0  \ 
            //     /  0    \ 
            //  1  --------- 2

            m = single_triangle();
        }
        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        executor.split_edge();

        const SimplicialComplex& simplices_to_delete = executor.simplices_to_delete;
        REQUIRE(simplices_to_delete.get_simplices().size() == 2);
        REQUIRE(simplices_to_delete.get_vertices().size() == 0);

        REQUIRE(simplices_to_delete.get_edges().size() == 1);
        // REQUIRE(executor.simplices_to_delete[1][0] == m._debug_id(edge, PE));
        REQUIRE(simplices_to_delete.get_faces().size() == 1);
        // REQUIRE(executor.simplices_to_delete[2][0] == 0);
    }
    SECTION("interior_edge")
    {
        // old faces are not recycled
        DEBUG_TriMesh m;
        {
            //  3--1--- 0
            //   |     / \ 
            //   2 f1 /2   1
            //   |  0/ f0  \ 
            //   |  /  0    \ 
            //  1  -------- 2
            //     \   1    /
            //      \  f2  /
            //       2    0
            //        \  /
            //         4
            RowVectors3l tris;
            tris.resize(3, 3);
            tris.row(0) = Eigen::Matrix<long, 3, 1>{0, 1, 2};
            tris.row(1) = Eigen::Matrix<long, 3, 1>{3, 1, 0};
            tris.row(2) = Eigen::Matrix<long, 3, 1>{1, 4, 2};
            m.initialize(tris);
        }
        REQUIRE(m.is_connectivity_valid());
        Tuple edge = m.edge_tuple_between_v1_v2(1, 2, 0);
        auto executor = m.get_tmoe(edge);

        executor.split_edge();

        const SimplicialComplex& simplices_to_delete = executor.simplices_to_delete;

        REQUIRE(simplices_to_delete.get_simplices().size() == 3);
        REQUIRE(simplices_to_delete.get_vertices().size() == 0);

        REQUIRE(simplices_to_delete.get_edges().size() == 1);
        // REQUIRE(simplices_to_delete[1][0] == m._debug_id(edge, PE));
        REQUIRE(simplices_to_delete.get_faces().size() == 2);
        // REQUIRE(simplices_to_delete[2][0] == 0);
        // REQUIRE(simplices_to_delete[2][1] == 2);
    }
}

TEST_CASE("split_edge", "[operations][2D]")
{
    //    0---1---2
    //   / \ / \ / \ .
    //  3---4---5---6
    //   \ / \ /
    //    7---8
    DEBUG_TriMesh m = hex_plus_two();

    REQUIRE(m.is_connectivity_valid());

    Tuple edge = m.edge_tuple_between_v1_v2(4, 5, 2);
    m.split_edge(edge);
    REQUIRE(m.is_connectivity_valid());

    Tuple edge2 = m.edge_tuple_between_v1_v2(3, 0, 0);
    m.split_edge(edge2);
    REQUIRE(m.is_connectivity_valid());

    Tuple edge3 = m.edge_tuple_between_v1_v2(4, 7, 6);
    m.split_edge(edge3);
    REQUIRE(m.is_connectivity_valid());

    Tuple edge4 = m.edge_tuple_between_v1_v2(4, 9, 8);
    m.split_edge(edge4);
    REQUIRE(m.is_connectivity_valid());

    Tuple edge5 = m.edge_tuple_between_v1_v2(5, 6, 4);
    m.split_edge(edge5);
    REQUIRE(m.is_connectivity_valid());
}

//////////// COLLAPSE TESTS ////////////
TEST_CASE("2D_link_condition_for_collapse", "[operations][2D][.]")
{
    REQUIRE(false);
}

TEST_CASE("collapse_edge", "[operations][2D][.]")
{
    DEBUG_TriMesh m;
    {
        //    0---1---2
        //   / \ / \ / \
            //  3---4---5---6
        //   \ / \ /
        //    7---8
        RowVectors3l tris;
        tris.resize(8, 3);
        tris << 3, 4, 0, 4, 1, 0, 4, 5, 1, 5, 2, 1, 5, 6, 2, 3, 7, 4, 7, 8, 4, 4, 8, 5;
        m.initialize(tris);
    }

    SECTION("case1")
    {
        std::cout << "BEFORE COLLAPSE" << std::endl;
        REQUIRE(m.is_connectivity_valid());

        Tuple edge = m.edge_tuple_between_v1_v2(4, 5, 2);
        m.collapse_edge(edge);
        std::cout << "AFTER COLLAPSE" << std::endl;
        REQUIRE(m.is_connectivity_valid());

        auto fv_accessor = m.create_base_accessor<long>(m.f_handle(PV));
        auto executor = m.get_tmoe(edge);

        REQUIRE(executor.flag_accessors[2].scalar_attribute(m.tuple_from_face_id(2)) == 0);
        REQUIRE(executor.flag_accessors[2].scalar_attribute(m.tuple_from_face_id(7)) == 0);
        REQUIRE(fv_accessor.vector_attribute(0)(1) == 9);
        REQUIRE(fv_accessor.vector_attribute(1)(0) == 9);
        REQUIRE(fv_accessor.vector_attribute(3)(0) == 9);
        REQUIRE(fv_accessor.vector_attribute(5)(2) == 9);
        REQUIRE(fv_accessor.vector_attribute(6)(2) == 9);
        REQUIRE(fv_accessor.vector_attribute(4)(0) == 9);
    }
    SECTION("case2")
    {
        std::cout << "BEFORE COLLAPSE" << std::endl;
        REQUIRE(m.is_connectivity_valid());

        Tuple edge = m.edge_tuple_between_v1_v2(0, 4, 0);
        m.collapse_edge(edge);
        std::cout << "AFTER COLLAPSE" << std::endl;
        REQUIRE(m.is_connectivity_valid());

        auto fv_accessor = m.create_base_accessor<long>(m.f_handle(PV));
        auto executor = m.get_tmoe(edge);

        REQUIRE(executor.flag_accessors[2].scalar_attribute(m.tuple_from_face_id(0)) == 0);
        REQUIRE(executor.flag_accessors[2].scalar_attribute(m.tuple_from_face_id(1)) == 0);

        REQUIRE(fv_accessor.vector_attribute(2)(0) == 9);
        REQUIRE(fv_accessor.vector_attribute(5)(2) == 9);
        REQUIRE(fv_accessor.vector_attribute(6)(2) == 9);
        REQUIRE(fv_accessor.vector_attribute(7)(0) == 9);
    }
}

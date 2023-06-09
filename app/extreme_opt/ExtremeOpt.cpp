#include "ExtremeOpt.h"
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/predicates/predicates.h>
#include <igl/write_triangle_mesh.h>
#include <tbb/concurrent_vector.h>
#include <wmtk/utils/AMIPS2D.h>
#include <Eigen/Core>

#include "SYMDIR_NEW.h"
#include <paraviewo/HDF5VTUWriter.hpp>
#include <paraviewo/ParaviewWriter.hpp>
#include <paraviewo/VTUWriter.hpp>
namespace extremeopt {

bool ExtremeOpt::has_degenerate_tris(const std::vector<Tuple>& tris) const
{
    for (const Tuple& new_tri : tris) {
        auto vids = this->oriented_tri_vids(new_tri);
        const auto [ai, bi, ci] = vids;
        assert(ai != bi);
        assert(bi != ci);
        assert(ai != ci);

        {
            const auto a = vertex_attrs[ai].pos;
            const auto b = vertex_attrs[bi].pos;
            const auto c = vertex_attrs[ci].pos;
            if (a == b || b == c || a == c) {
                spdlog::warn("DEGENERATE CASE FOUND");
                return true;
            }
        }
        {
            const auto a = vertex_attrs[ai].pos_3d;
            const auto b = vertex_attrs[bi].pos_3d;
            const auto c = vertex_attrs[ci].pos_3d;
            if (a == b || b == c || a == c) {
                spdlog::warn("DEGENERATE CASE FOUND");
                return true;
            }
        }
    }
    return false;
}
void ExtremeOpt::create_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv)
{
    // init aabb tree
    tree.init(V, F);
    input_V = V;
    input_F = F;
    
    // Register attributes
    p_vertex_attrs = &vertex_attrs;
    p_face_attrs = &face_attrs;

    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);

    // Convert from eigen to internal representation (TODO: move to utils and remove it from all
    // app)
    std::vector<std::array<size_t, 3>> tri(F.rows());

    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            tri[i][j] = (size_t)F(i, j);
        }
    }

    // Initialize the trimesh class which handles connectivity
    wmtk::TriMesh::create_mesh(V.rows(), tri);

    // TODO: this is possibly not needed
    // Save the face area in the face attributes
    for (int i = 0; i < F.rows(); i++) {
        face_attrs[i].area_3d = dblarea[i];
    }

    // Save the vertex position in the vertex attributes
    for (unsigned i = 0; i < V.rows(); ++i) {
        vertex_attrs[i].pos = uv.row(i).transpose();
        vertex_attrs[i].pos_3d = V.row(i).transpose();
    }
    std::vector<Eigen::Vector3d> V_in(V.rows());
    std::vector<Eigen::Vector3i> F_in(F.rows());
    for (auto i = 0; i < V.rows(); i++) {
        V_in[i] = vertex_attrs[i].pos_3d;
    }
    for (int i = 0; i < F_in.size(); ++i) F_in[i] << F(i,0), F(i,1), F(i,2); 
    
    double diag = std::sqrt(std::pow(V.col(0).maxCoeff() - V.col(0).minCoeff(), 2)+std::pow(V.col(1).maxCoeff() - V.col(1).minCoeff(), 2)+std::pow(V.col(2).maxCoeff() - V.col(2).minCoeff(), 2));
    if (this->m_params.use_envelope)
    {
        m_envelope.use_exact = false;
        m_envelope.init(V_in, F_in, 0.01 * diag);
    }

    // std::vector<std::vector<int>> bds;
    // igl::boundary_loop(F, bds);
    // for (auto bd : bds) {
    //     for (int vec : bd) {
    //         vertex_attrs[vec].fixed = true;
    //     }
    // }
}

void ExtremeOpt::init_constraints(const std::vector<std::vector<int>>& EE_e)
{
    p_edge_attrs = &edge_attrs;
    edge_attrs.resize(tri_capacity() * 3);
    for (int i = 0; i < EE_e.size(); i++) {
        auto t1 = tuple_from_edge(EE_e[i][0], EE_e[i][1]);
        auto t2 = tuple_from_edge(EE_e[i][2], EE_e[i][3]);
        int eid1 = t1.eid(*this);
        int eid2 = t2.eid(*this);
        // std::cout << "(" << eid1 << ", " << eid2 << ")" << is_boundary_edge(t1) <<
        // is_boundary_edge(t2) << std::endl;

        edge_attrs[eid1].pair = t2;
        edge_attrs[eid2].pair = t1;
    }
}

void ExtremeOpt::update_constraints_EE_v(const Eigen::MatrixXi& EE)
{
    edge_attrs.resize(tri_capacity() * 3);
    for (int i = 0; i < EE.rows(); i++) {
        int v0 = EE(i, 0);
        int v1 = EE(i, 1);
        int v2 = EE(i, 2);
        int v3 = EE(i, 3);

        auto t = tuple_from_vertex(v0);
        auto one_ring_v0 = get_one_ring_edges_for_vertex(t);
        t = tuple_from_vertex(v2);
        auto one_ring_v2 = get_one_ring_edges_for_vertex(t);
        Tuple t_e01, t_e23;
        for (auto t_tmp : one_ring_v0) {
            if (t_tmp.vid(*this) == v1) {
                t_e01 = t_tmp.switch_vertex(*this);
            }
        }
        for (auto t_tmp : one_ring_v2) {
            if (t_tmp.vid(*this) == v3) {
                t_e23 = t_tmp.switch_vertex(*this);
            }
        }

        edge_attrs[t_e01.eid(*this)].pair = t_e23;
        edge_attrs[t_e23.eid(*this)].pair = t_e01;
    }
}

void ExtremeOpt::export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv)
{
    // consolidate_mesh();
    consolidate_mesh_cons(); // use the one with constraints
    V = Eigen::MatrixXd::Zero(vert_capacity(), 3);
    uv = Eigen::MatrixXd::Zero(vert_capacity(), 2);
    for (auto& t : get_vertices()) {
        auto i = t.vid(*this);
        auto mv = V.row(i) = vertex_attrs[i].pos_3d;
        auto muv = uv.row(i) = vertex_attrs[i].pos;

        if (!mv.array().isFinite().all()) {
            spdlog::warn("mv {} is not finite: {}", i, fmt::join(mv, ","));
            ;
        }
        if (!muv.array().isFinite().all()) {
            spdlog::warn("muv {} is not finite: {}", i, fmt::join(muv, ","));
            ;
        }
    }

    F = Eigen::MatrixXi::Constant(tri_capacity(), 3, -1);
    for (auto& t : get_faces()) {
        auto i = t.fid(*this);
        auto vs = oriented_tri_vertices(t);
        for (int j = 0; j < 3; j++) {
            F(i, j) = vs[j].vid(*this);
        }
    }
}

void ExtremeOpt::export_mesh_vtu(const std::string &dir, const std::string &filename)
{
    std::vector<Eigen::RowVector3d> V_vec;
    std::vector<Eigen::RowVector2d> uv_vec;

    auto v_cnt = 0;
    std::vector<size_t> map_v_ids(vert_capacity(), -1);
    for (auto i = 0; i < vert_capacity(); i++) {
        if (m_vertex_connectivity[i].m_is_removed) continue;
        map_v_ids[i] = v_cnt;
        V_vec.push_back(vertex_attrs[i].pos_3d);
        uv_vec.push_back(vertex_attrs[i].pos);
        v_cnt++;
    }

    std::vector<Eigen::RowVector3i> F_vec;
    std::vector<int> face_ids_vec;
    auto t_cnt = 0;
    std::vector<size_t> map_t_ids(tri_capacity(), -1);
    for (auto i = 0; i < tri_capacity(); i++) {
        if (m_tri_connectivity[i].m_is_removed) continue;
        map_t_ids[i] = t_cnt;
        Eigen::RowVector3i face;
        face << map_v_ids[m_tri_connectivity[i][0]], map_v_ids[m_tri_connectivity[i][1]], map_v_ids[m_tri_connectivity[i][2]];
        F_vec.push_back(face);
        face_ids_vec.push_back(i);
        t_cnt++;
    }

    Eigen::MatrixXd V(v_cnt, 3);
    Eigen::MatrixXd uv(v_cnt, 2);
    for (auto i = 0; i < v_cnt; i++)
    {
        V.row(i) = V_vec[i];
        uv.row(i) = uv_vec[i];
    }
    Eigen::MatrixXi F(t_cnt, 3);
    for (auto i = 0; i < t_cnt; i++)
    {
        F.row(i) = F_vec[i];
    }


    paraviewo::HDF5VTUWriter writer;
    auto Es = get_quality_all();
    Eigen::MatrixXd dirichlet_energy;
    dirichlet_energy.resize(F.rows(), 1);
    for (unsigned i = 0; i < F.rows(); ++i) {
        dirichlet_energy(i, 0) = Es(face_ids_vec[i]);
    }
    Eigen::MatrixXd face_id;
    face_id.resize(F.rows(), 1);
    for (unsigned i = 0; i < F.rows(); ++i){
        face_id(i, 0) = face_ids_vec[i];
    }

    writer.add_cell_field("dirichlet_energy", dirichlet_energy);
    writer.add_cell_field("face_id", face_id);
    writer.write_mesh(dir + filename, V, F);
    writer.write_mesh(dir + "param_" + filename, uv, F);
}   

void ExtremeOpt::export_EE(Eigen::MatrixXi& EE)
{
    EE.resize(0, 0);
    for (auto& loc : get_edges()) {
        if (is_boundary_edge(loc)) {
            EE.conservativeResize(EE.rows() + 1, 4);
            EE.row(EE.rows() - 1) << loc.vid(*this), loc.switch_vertex(*this).vid(*this),
                edge_attrs[loc.eid(*this)].pair.vid(*this),
                edge_attrs[loc.eid(*this)].pair.switch_vertex(*this).vid(*this);
        }
    }
}

void ExtremeOpt::write_obj(const std::string& path)
{
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;

    export_mesh(V, F, uv);

    igl::writeOBJ(path, V, F, V, F, uv, F);
}


double ExtremeOpt::get_quality()
{
    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
    auto vs = get_vertices();
    double energy = 0;
    for (auto loc : vs)
    {
        energy += E_eval.symmetric_dirichlet_energy_onering(*this, loc);
    }

    return energy / 3.0;
}

double ExtremeOpt::get_quality_max()
{
    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Max, 1);
    auto vs = get_vertices();
    double energy = 0;
    for (auto loc : vs)
    {
        energy = std::max(energy, E_eval.symmetric_dirichlet_energy_onering(*this, loc));
    }

    return energy;
}

Eigen::VectorXd ExtremeOpt::get_quality_all()
{
    Eigen::VectorXd Es(m_tri_connectivity.size());
    Es.setZero();
    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
    auto fs = get_faces();
    for (auto loc : fs)
    {
        auto local_tuples = oriented_tri_vertices(loc);
        Eigen::Vector3d A = vertex_attrs[local_tuples[0].vid(*this)].pos_3d;
        Eigen::Vector3d B = vertex_attrs[local_tuples[1].vid(*this)].pos_3d;
        Eigen::Vector3d C = vertex_attrs[local_tuples[2].vid(*this)].pos_3d;
        Eigen::Vector2d a = vertex_attrs[local_tuples[0].vid(*this)].pos;
        Eigen::Vector2d b = vertex_attrs[local_tuples[1].vid(*this)].pos;
        Eigen::Vector2d c = vertex_attrs[local_tuples[2].vid(*this)].pos;
        Es(loc.fid(*this)) = E_eval.symmetric_dirichlet_energy(A, B, C, a, b, c);
    }
    return Es;
}

int ExtremeOpt::get_mesh_onering(
    const Tuple& t,
    Eigen::MatrixXd& V_local,
    Eigen::MatrixXd& uv_local,
    Eigen::MatrixXi& F_local)
{
    auto vid = t.vid(*this);
    auto vid_onering = get_one_ring_vids_for_vertex(vid);
    auto locs = get_one_ring_tris_for_vertex(t);
    V_local.resize(vid_onering.size(), 3);
    uv_local.resize(vid_onering.size(), 2);
    for (size_t i = 0; i < vid_onering.size(); i++) {
        V_local.row(i) = vertex_attrs[vid_onering[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[vid_onering[i]].pos;
    }
    std::vector<int> v_map(vertex_attrs.size(), -1);
    for (size_t i = 0; i < vid_onering.size(); i++) {
        v_map[vid_onering[i]] = i;
    }
    F_local.resize(locs.size(), 3);
    for (size_t i = 0; i < locs.size(); i++) {
        int t_id = locs[i].fid(*this);
        auto local_tuples = oriented_tri_vertices(locs[i]);
        for (size_t j = 0; j < 3; j++) {
            F_local(i, j) = v_map[local_tuples[j].vid(*this)];
        }
    }
    return v_map[vid];
}

void ExtremeOpt::get_mesh_onering_edge(
    const Tuple& t,
    Eigen::MatrixXd& V_local,
    Eigen::MatrixXd& uv_local,
    Eigen::MatrixXi& F_local)
{

    auto vid1 = t.vid(*this);
    auto vid_onering1 = get_one_ring_vids_for_vertex(vid1);
    auto locs1 = get_one_ring_tris_for_vertex(t);

    auto vid2 = t.switch_vertex(*this).vid(*this);
    auto vid_onering2 = get_one_ring_vids_for_vertex(vid2);
    auto locs2 = get_one_ring_tris_for_vertex(t.switch_vertex(*this));

    int V_size = vid_onering1.size() + vid_onering2.size() - 2;
    int F_size = locs1.size() + locs2.size(); 
    if (is_boundary_edge(t))
    {
        V_size -= 1;
        F_size -= 1;
    }
    else
    {
        V_size -= 2;
        F_size -= 2;
    }

    V_local.resize(V_size, 3);
    uv_local.resize(V_size, 2);

    std::vector<int> v_map(vertex_attrs.size(), -1);
    for(int i = 0; i < vid_onering1.size(); i++)
    {
        v_map[vid_onering1[i]] = i;
        V_local.row(i) = vertex_attrs[vid_onering1[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[vid_onering1[i]].pos;
    }
    int cnt = vid_onering1.size();
    for (int i = 0; i < vid_onering2.size(); i++)
    {
        if (v_map[vid_onering2[i]] == -1)
        {
            v_map[vid_onering2[i]] = cnt;
            V_local.row(cnt) = vertex_attrs[vid_onering2[i]].pos_3d;
            uv_local.row(cnt) = vertex_attrs[vid_onering2[i]].pos;
            cnt++;
        }
    }
    
    if (cnt != V_size)
    {
        std::cout << "V_size Error in get_mesh_one_ring_edge" << std::endl;
    }

    F_local.resize(F_size, 3);
    std::vector<bool> is_f_used(m_tri_connectivity.size(), false);
    for (int i = 0; i < locs1.size(); i++) {
        int t_id = locs1[i].fid(*this);
        is_f_used[t_id] = true;
        auto local_tuples = oriented_tri_vertices(locs1[i]);
        for (int j = 0; j < 3; j++) {
            F_local(i, j) = v_map[local_tuples[j].vid(*this)];
        }
    }
    int f_cnt = locs1.size();
    for (int i = 0; i < locs2.size(); i++){
        int t_id = locs2[i].fid(*this);
        if (!is_f_used[t_id])
        {
            is_f_used[t_id] = true;
            auto local_tuples = oriented_tri_vertices(locs2[i]);
            for (int j = 0; j < 3; j++)
            {
                F_local(f_cnt, j) = v_map[local_tuples[j].vid(*this)];
            }
            f_cnt++;
        }
    }

}


bool ExtremeOpt::is_inverted(const Tuple& loc) const
{
    // Get the vertices ids
    auto vs = oriented_tri_vertices(loc);

    igl::predicates::exactinit();

    // Use igl for checking orientation
    auto res = igl::predicates::orient2d(
        vertex_attrs[vs[0].vid(*this)].pos,
        vertex_attrs[vs[1].vid(*this)].pos,
        vertex_attrs[vs[2].vid(*this)].pos);

    // The element is inverted if it not positive (i.e. it is negative or it is degenerate)
    return (res != igl::predicates::Orientation::POSITIVE);
}

bool ExtremeOpt::is_3d_degenerated(const Tuple& loc) const
{
    // Get the vertices ids
    auto vs = oriented_tri_vertices(loc);

    Eigen::Vector3d A = vertex_attrs[vs[0].vid(*this)].pos_3d;
    Eigen::Vector3d B = vertex_attrs[vs[1].vid(*this)].pos_3d;
    Eigen::Vector3d C = vertex_attrs[vs[2].vid(*this)].pos_3d;

    double area = ((B-A).cross(C-A)).norm();
    return (area <= 0);
}



void ExtremeOpt::consolidate_mesh_cons()
{
    auto v_cnt = 0;
    std::vector<size_t> map_v_ids(vert_capacity(), -1);
    for (auto i = 0; i < vert_capacity(); i++) {
        if (m_vertex_connectivity[i].m_is_removed) continue;
        map_v_ids[i] = v_cnt;
        v_cnt++;
    }

    auto t_cnt = 0;
    std::vector<size_t> map_t_ids(tri_capacity(), -1);
    for (auto i = 0; i < tri_capacity(); i++) {
        if (m_tri_connectivity[i].m_is_removed) continue;
        map_t_ids[i] = t_cnt;
        t_cnt++;
    }
    v_cnt = 0;
    for (auto i = 0; i < vert_capacity(); i++) {
        if (m_vertex_connectivity[i].m_is_removed) continue;
        if (v_cnt != i) {
            assert(v_cnt < i);
            m_vertex_connectivity[v_cnt] = m_vertex_connectivity[i];
            if (p_vertex_attrs) p_vertex_attrs->move(i, v_cnt);
        }
        for (size_t& t_id : m_vertex_connectivity[v_cnt].m_conn_tris) t_id = map_t_ids[t_id];
        v_cnt++;
    }
    t_cnt = 0;

    for (int i = 0; i < tri_capacity(); i++) {
        if (m_tri_connectivity[i].m_is_removed) continue;

        if (t_cnt != i) {
            assert(t_cnt < i);
            m_tri_connectivity[t_cnt] = m_tri_connectivity[i];
            m_tri_connectivity[t_cnt].hash = 0;
            if (p_face_attrs) p_face_attrs->move(i, t_cnt);

            for (auto j = 0; j < 3; j++) {
                if (p_edge_attrs) p_edge_attrs->move(i * 3 + j, t_cnt * 3 + j);
            }
        }
        for (size_t& v_id : m_tri_connectivity[t_cnt].m_indices) v_id = map_v_ids[v_id];
        t_cnt++;
    }

    current_vert_size = v_cnt;
    current_tri_size = t_cnt;

    m_vertex_connectivity.m_attributes.resize(v_cnt);
    m_vertex_connectivity.shrink_to_fit();
    m_tri_connectivity.m_attributes.resize(t_cnt);
    m_tri_connectivity.shrink_to_fit();

    resize_mutex(vert_capacity());

    // Resize user class attributes
    if (p_vertex_attrs) p_vertex_attrs->resize(vert_capacity());
    if (p_edge_attrs) p_edge_attrs->resize(tri_capacity() * 3);
    if (p_face_attrs) p_face_attrs->resize(tri_capacity());

    if (m_params.with_cons)
    {
        // update constraints(edge tuple pairs)
        for (int i = 0; i < tri_capacity(); i++) {
            for (int j = 0; j < 3; j++) {
                auto cur_t = tuple_from_edge(i, j);
                if (is_boundary_edge(cur_t)) {
                    auto pair_t = edge_attrs[3 * i + j].pair;
                    edge_attrs[3 * i + j].pair = tuple_from_edge(
                        map_t_ids[pair_t.eid_unsafe(*this) / 3],
                        pair_t.eid_unsafe(*this) % 3);
                }
            }
        }
    }
}

bool ExtremeOpt::invariants(const wmtk::TriMeshOperation& op) {
    return invariants(op.modified_triangles(*this));
}
bool ExtremeOpt::invariants(const std::vector<Tuple>& new_tris)
{
    if (m_params.use_envelope)
    {
        for (auto& t : new_tris) {
            std::array<Eigen::Vector3d, 3> tris;
            auto vs = oriented_tri_vertices(t);
            for (auto j = 0; j < 3; j++) tris[j] = vertex_attrs[vs[j].vid(*this)].pos_3d;
            if (m_envelope.is_outside(tris)) {
                return false;
            }
        }
    } 
    
    return true;
}

bool ExtremeOpt::check_constraints(double eps)
{
    auto all_edges = this->get_edges();
    bool flag = true;
    for (auto t_e : all_edges) {
        if (!this->is_boundary_edge(t_e)) continue;
        auto t_e_pair = edge_attrs[t_e.eid(*this)].pair;
        int v0 = t_e.vid(*this);
        int v1 = t_e.switch_vertex(*this).vid(*this);
        int v2 = t_e_pair.vid(*this);
        int v3 = t_e_pair.switch_vertex(*this).vid(*this);

        // check length
        auto e_ab = (vertex_attrs[v1].pos - vertex_attrs[v0].pos);
        auto e_dc = (vertex_attrs[v2].pos - vertex_attrs[v3].pos);
        // std::cout << "(" << v0 << "," << v1 << ") - (" << v2 << "," << v3 << ")" << std::endl;
        // std::cout << abs(e_ab.norm() - e_dc.norm()) << std::endl;
        if (abs(e_ab.norm() - e_dc.norm()) > eps) {
            std::cout << "length error "
                      << "(" << v0 << "," << v1 << ") - (" << v2 << "," << v3 << ")" << std::endl;
            std::cout << abs(e_ab.norm() - e_dc.norm()) << std::endl;
            flag = false;
            // return false;
        }

        // check angle
        Eigen::Vector2d e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        double index = 2 * angle / igl::PI;
        // std::cout << index << std::endl;
        if (abs(index - round(index)) > eps) {
            std::cout << "angle error "
                      << "(" << v0 << "," << v1 << ") - (" << v2 << "," << v3 << ")" << std::endl;
            std::cout << index << std::endl;
            flag = false;
            // return false;
        }
    }

    return flag;
    // return true;
}

} // namespace extremeopt

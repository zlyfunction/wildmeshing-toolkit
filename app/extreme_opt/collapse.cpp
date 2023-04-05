#include "ExtremeOpt.h"
#include "SYMDIR.h"

bool extremeopt::ExtremeOpt::collapse_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) {
        return false;
    }
    if (!wmtk::TriMesh::collapse_edge_before(t)) {
        return false;
    }

    // avoid boundary edges here
    if (m_params.with_cons && is_boundary_edge(t))
    {
        std::cout << "Error: Wrong function call when collapsing a boudnary edge" << std::endl;
        return false;
    }

    cache_edge_positions(t);
    return true;
}

bool extremeopt::ExtremeOpt::collapse_edge_after(const Tuple& t)
{
    // const Eigen::Vector3d V = (position_cache.local().V1 + position_cache.local().V2) / 2.0;
    // const Eigen::Vector2d uv = (position_cache.local().uv1 + position_cache.local().uv2) / 2.0;
    Eigen::Vector3d V;
    Eigen::Vector2d uv;
    if (position_cache.local().is_v1_bd) {
        V = position_cache.local().V1;
        uv = position_cache.local().uv1;
    } else {
        V = position_cache.local().V2;
        uv = position_cache.local().uv2;
    }
    // const Eigen::Vector3d V = position_cache.local().V1;
    // const Eigen::Vector2d uv = position_cache.local().uv1;

    auto vid = t.vid(*this);
    auto& v = vertex_attrs[vid];
    v.pos_3d = V;
    v.pos = uv;

    // get local F,V,uv
    auto vid_onering = get_one_ring_vids_for_vertex(vid);

    // check edgelen
    for (int vid_tmp : vid_onering)
    {
        auto V_tmp = vertex_attrs[vid_tmp].pos_3d;
        auto uv_tmp = vertex_attrs[vid_tmp].pos;
        double elen_3d = (V_tmp - V).norm();
        double elen = (uv_tmp - uv).norm();
        if (elen > elen_threshold) return false;
        if (elen_3d > elen_threshold_3d) return false;
    }

    auto locs = get_one_ring_tris_for_vertex(t);
    Eigen::MatrixXd V_local(vid_onering.size(), 3);
    Eigen::MatrixXd uv_local(vid_onering.size(), 2);
    for (size_t i = 0; i < vid_onering.size(); i++) {
        V_local.row(i) = vertex_attrs[vid_onering[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[vid_onering[i]].pos;
    }
    std::vector<int> v_map(vertex_attrs.size(), -1);
    for (size_t i = 0; i < vid_onering.size(); i++) {
        v_map[vid_onering[i]] = i;
    }
    Eigen::MatrixXi F_local(locs.size(), 3);
    for (size_t i = 0; i < locs.size(); i++) {
        int t_id = locs[i].fid(*this);
        auto local_tuples = oriented_tri_vertices(locs[i]);
        for (size_t j = 0; j < 3; j++) {
            F_local(i, j) = v_map[local_tuples[j].vid(*this)];
        }
    }
    Eigen::VectorXd area_local_3d, area_local;
    igl::doublearea(V_local, F_local, area_local_3d);
    igl::doublearea(uv_local, F_local, area_local);


    // export_mesh(V_local, F_local, uv_local);
    // first check flips
    if (area_local_3d.minCoeff() <= 0 || area_local.minCoeff() <= 0) {
        // std::cout << "collapse causing flips" << std::endl;
        return false;
    }

    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    double E_after_collapse = wmtk::compute_energy_from_jacobian(Ji, area_local_3d) * area_local_3d.sum();

    std::cout << "E_after collapse: " << E_after_collapse  << " area: " << area_local_3d.sum() << std::endl;
    if (E_after_collapse > position_cache.local().E_before_collapse) {
        // E_sum does not go down
        return false;
    }
    std::cout << "collapse good" << std::endl;

    if (m_params.with_cons)
    {
        // update constraints
        if (this->is_boundary_vertex(t)) {
            // std::cout << "update constraints around vertex " << t.vid(*this) << std::endl;

            auto one_ring_edges = this->get_one_ring_edges_for_vertex(t);
            std::vector<Tuple> two_bd_e;
            Tuple e_old_l, e_old_r;
            Tuple e_new_l, e_new_r;
            for (auto e : one_ring_edges) {
                if (this->is_boundary_edge(e)) {
                    Tuple candidate_e = e;
                    if (!candidate_e.is_ccw(*this)) {
                        candidate_e = candidate_e.switch_vertex(*this);
                    }
                    if (candidate_e.vid(*this) == t.vid(*this)) {
                        e_new_r = candidate_e;
                    } else {
                        e_new_l = candidate_e;
                    }
                }
            }
            if (position_cache.local().bd_e1.vid(*this) == position_cache.local().vid1 ||
                position_cache.local().bd_e1.vid(*this) == position_cache.local().vid2) {
                e_old_r = position_cache.local().bd_e1;
                e_old_l = position_cache.local().bd_e2;
            } else {
                e_old_r = position_cache.local().bd_e2;
                e_old_l = position_cache.local().bd_e1;
            }

            auto e_old_r_pair = edge_attrs[e_old_r.eid_unsafe(*this)].pair;
            auto e_old_l_pair = edge_attrs[e_old_l.eid_unsafe(*this)].pair;

            if (edge_attrs[e_old_r.eid_unsafe(*this)].pair.eid_unsafe(*this) ==
                e_old_l.eid_unsafe(*this)) {
                // std::cout << "in this case the two new bd edges is a pair" << std::endl;
                edge_attrs[e_new_r.eid(*this)].pair = e_new_l;
                edge_attrs[e_new_l.eid(*this)].pair = e_new_r;
            } else {
                edge_attrs[e_old_r_pair.eid_unsafe(*this)].pair = e_new_r;
                edge_attrs[e_old_l_pair.eid_unsafe(*this)].pair = e_new_l;
                edge_attrs[e_new_r.eid(*this)].pair = e_old_r_pair;
                edge_attrs[e_new_l.eid(*this)].pair = e_old_l_pair;
            }
        }
        auto one_ring_tris = get_one_ring_tris_for_vertex(t);

        for (auto t_tmp : one_ring_tris) {
            Tuple t0 = t_tmp;
            Tuple t1 = t0.switch_edge(*this);
            Tuple t2 = t1.switch_vertex(*this).switch_edge(*this);
            if (!t0.is_ccw(*this)) t0 = t0.switch_vertex(*this);
            if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
            if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);

            if (is_boundary_edge(t0)) {
                edge_attrs[edge_attrs[t0.eid(*this)].pair.eid_unsafe(*this)].pair = t0;
            }
            if (is_boundary_edge(t1)) {
                edge_attrs[edge_attrs[t1.eid(*this)].pair.eid_unsafe(*this)].pair = t1;
            }
            if (is_boundary_edge(t2)) {
                edge_attrs[edge_attrs[t2.eid(*this)].pair.eid_unsafe(*this)].pair = t2;
            }
        }
    }

    return true;
}

bool extremeopt::ExtremeOpt::collapse_bd_edge_after(
    const Tuple& t,
    const Eigen::Vector3d& V_keep,
    const Eigen::Vector2d& uv_keep,
    Tuple& t_l_old,
    Tuple& t_r_old,
    double& E)
{
    // update vertex position
    auto vid = t.vid(*this);
    vertex_attrs[vid].pos_3d = V_keep;
    vertex_attrs[vid].pos = uv_keep;

    // get local mesh and check area/E_max
    Eigen::MatrixXd V_local, uv_local;
    Eigen::MatrixXi F_local;
    get_mesh_onering(t, V_local, uv_local, F_local);
    Eigen::VectorXd area_local_3d, area_local;
    igl::doublearea(V_local, F_local, area_local_3d);
    igl::doublearea(uv_local, F_local, area_local);
    if (area_local_3d.minCoeff() <= 0 || area_local.minCoeff() <= 0) {
        return false;
    }
    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    auto Es = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    for (int i = 0; i < Es.size(); i++)
    {
        if (!std::isfinite(Es(i)))
        {
            return false;
        }
    }

    // compute energy here
    E = Es.dot(area_local_3d);

    // check envelope
    if (!invariants(get_one_ring_tris_for_vertex(t)))
    {
        return false;
    }
    // update constraints
    auto one_ring_edges = this->get_one_ring_edges_for_vertex(t);
    Tuple e_new_l, e_new_r;
    for (auto e : one_ring_edges) {
        if (this->is_boundary_edge(e)) {
            Tuple candidate_e = e;
            if (!candidate_e.is_ccw(*this)) {
                candidate_e = candidate_e.switch_vertex(*this);
            }
            if (candidate_e.vid(*this) == t.vid(*this)) {
                e_new_r = candidate_e;
            } else {
                e_new_l = candidate_e;
            }
        }
    }
    auto e_old_l_pair = edge_attrs[t_l_old.eid_unsafe(*this)].pair;
    auto e_old_r_pair = edge_attrs[t_r_old.eid_unsafe(*this)].pair;
    edge_attrs[e_old_r_pair.eid_unsafe(*this)].pair = e_new_r;
    edge_attrs[e_old_l_pair.eid_unsafe(*this)].pair = e_new_l;
    edge_attrs[e_new_r.eid(*this)].pair = e_old_r_pair;
    edge_attrs[e_new_l.eid(*this)].pair = e_old_l_pair;

    auto one_ring_tris = get_one_ring_tris_for_vertex(t);
    for (auto t_tmp : one_ring_tris) {
        Tuple t0 = t_tmp;
        Tuple t1 = t0.switch_edge(*this);
        Tuple t2 = t1.switch_vertex(*this).switch_edge(*this);
        if (!t0.is_ccw(*this)) t0 = t0.switch_vertex(*this);
        if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
        if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);

        if (is_boundary_edge(t0)) {
            edge_attrs[edge_attrs[t0.eid(*this)].pair.eid_unsafe(*this)].pair = t0;
        }
        if (is_boundary_edge(t1)) {
            edge_attrs[edge_attrs[t1.eid(*this)].pair.eid_unsafe(*this)].pair = t1;
        }
        if (is_boundary_edge(t2)) {
            edge_attrs[edge_attrs[t2.eid(*this)].pair.eid_unsafe(*this)].pair = t2;
        }
    }


    return true;
}
#include "ExtremeOpt.h"
#include "SYMDIR.h"
bool extremeopt::ExtremeOpt::swap_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) return false;
    if (!TriMesh::swap_edge_before(t)) return false;

    Tuple t1 = t.switch_vertex(*this).switch_edge(*this);
    Tuple t2 = t.switch_face(*this).value().switch_edge(*this);
    if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
    if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);
    swap_cache.local().t1 = t1;
    swap_cache.local().t2 = t2;

    // compute old local energy
    auto tri1 = oriented_tri_vids(t);
    auto t_opp = t.switch_face(*this);
    auto tri2 = oriented_tri_vids(t_opp.value());
    
    std::vector<int> v_ids;
    std::vector<int> v_map(vertex_attrs.size());
    Eigen::MatrixXi F_local(2, 3);
    Eigen::MatrixXd V_local(4, 3), uv_local(4, 2);
    for (int i = 0; i < 3; i++)
    {
        v_ids.push_back((int)tri1[i]);
    }
    for (int i = 0; i < 3; i++)
    {
        if (std::find(v_ids.begin(), v_ids.end(), (int)tri2[i]) == v_ids.end())
        {
            v_ids.push_back((int)tri2[i]);
        }
    }
    std::sort(v_ids.begin(), v_ids.end());
    for (int i = 0; i < v_ids.size(); i++)
    {
        v_map[v_ids[i]] = i;
    }
    F_local.row(0) << v_map[tri1[0]], v_map[tri1[1]], v_map[tri1[2]];
    F_local.row(1) << v_map[tri2[0]], v_map[tri2[1]], v_map[tri2[2]];
    for (int i = 0; i < 4; i++) {
        V_local.row(i) = vertex_attrs[v_ids[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[v_ids[i]].pos;
    }
    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    swap_cache.local().E_old = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();

    return true;
}



bool extremeopt::ExtremeOpt::swap_edge_after(const Tuple& t)
{
    // std::cout << "connectivity test:" << std::endl;
    auto tri1 = oriented_tri_vids(t);
    auto t_opp = t.switch_face(*this);
    auto tri2 = oriented_tri_vids(t_opp.value());

    std::vector<int> v_ids;
    std::vector<int> v_map(vertex_attrs.size());
    Eigen::MatrixXi F_local(2, 3);
    Eigen::MatrixXd V_local(4, 3), uv_local(4, 2);
    for (int i = 0; i < 3; i++) {
        v_ids.push_back(tri1[i]);
    }
    for (int i = 0; i < 3; i++) {
        int j = 0;
        while (j < 3 && v_ids[j] != (int)tri2[i]) {
            j++;
        }
        if (j == 3) v_ids.push_back((int)tri2[i]);
    }
    std::sort(v_ids.begin(), v_ids.end());
    assert(v_ids.size() == 4);
    for (int i = 0; i < v_ids.size(); i++)
    {
        v_map[v_ids[i]] = i;
    }

    F_local.row(0) << v_map[tri1[0]], v_map[tri1[1]], v_map[tri1[2]];
    F_local.row(1) << v_map[tri2[0]], v_map[tri2[1]], v_map[tri2[2]];

    for (int i = 0; i < 4; i++) {
        V_local.row(i) = vertex_attrs[v_ids[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[v_ids[i]].pos;
    }

    Eigen::VectorXd dblarea, dblarea_3d;
    igl::doublearea(V_local, F_local, dblarea_3d);
    igl::doublearea(uv_local, F_local, dblarea);


    if (dblarea_3d.minCoeff() <= 0) {
        return false;
    }
    if (dblarea.minCoeff() <= 0) {
        return false;
    }

    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);

    double E, E_old;
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    Eigen::VectorXd Es =
        wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    
    E = Es.maxCoeff(); // compute E_max
    if (!std::isfinite(Es(0)) || !std::isfinite(Es(1))) {
        // std::cout << "nan fail" << std::endl;
        return false;
    }
    if (Es.minCoeff() <= 0) {
        return false;
    }

    if (E >= swap_cache.local().E_old)
    {
        return false;
    }
    
    if (m_params.with_cons)
    {
        // update constraints after swap
        Tuple t1 = t.switch_edge(*this);
        Tuple t2 = t.switch_face(*this).value().switch_vertex(*this).switch_edge(*this);
        Tuple t1_old = swap_cache.local().t1;
        Tuple t2_old = swap_cache.local().t2;
        if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
        if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);
        bool flag = true;
        if (is_boundary_edge(t1)) {
            auto t1_old_pair = edge_attrs[t1_old.eid_unsafe(*this)].pair;
            if (t1_old_pair.eid_unsafe(*this) == t2_old.eid_unsafe(*this)) {
                flag = false;
                edge_attrs[t1.eid(*this)].pair = t2;
                edge_attrs[t2.eid(*this)].pair = t1;
            } else {
                edge_attrs[t1_old_pair.eid_unsafe(*this)].pair = t1;
                edge_attrs[t1.eid(*this)].pair = t1_old_pair;
            }
        }
        if (flag && is_boundary_edge(t2)) {
            auto t2_old_pair = edge_attrs[t2_old.eid_unsafe(*this)].pair;
            edge_attrs[t2_old_pair.eid_unsafe(*this)].pair = t2;
            edge_attrs[t2.eid(*this)].pair = t2_old_pair;
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

void extremeopt::ExtremeOpt::swap_all_edges()
{
    auto collect_all_ops_swap = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_edges()) {
        collect_all_ops_swap.emplace_back("edge_swap", loc);
    }

    auto setup_and_execute = [&](auto& executor_swap) {
        executor_swap.renew_neighbor_tuples = renew;
        executor_swap.priority = [&](auto& m, auto _, auto& e) {
            return -(vertex_attrs[e.vid(*this)].pos -
                     vertex_attrs[e.switch_vertex(*this).vid(*this)].pos)
                        .norm();
        };
        executor_swap.num_threads = NUM_THREADS;
        executor_swap(*this, collect_all_ops_swap);
    };
    auto executor_swap = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    setup_and_execute(executor_swap);
}

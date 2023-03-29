#include "ExtremeOpt.h"
#include "SYMDIR.h"
bool extremeopt::ExtremeOpt::split_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) {
        return false;
    }
    if (!TriMesh::split_edge_before(t)) {
        return false;
    }


    if (is_boundary_edge(t)) {
        return false;
    }

    position_cache.local().V1 = vertex_attrs[t.vid(*this)].pos_3d;
    position_cache.local().V2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos_3d;
    position_cache.local().uv1 = vertex_attrs[t.vid(*this)].pos;
    position_cache.local().uv2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos;

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
    position_cache.local().E_max_before_collpase = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();
    return true;
}

bool extremeopt::ExtremeOpt::split_edge_after(const Tuple& t)
{
    Eigen::Vector3d V = (position_cache.local().V1 + position_cache.local().V2) / 2.0;
    Eigen::Vector2d uv = (position_cache.local().uv1 + position_cache.local().uv2) / 2.0;
    Tuple vert_tuple = t.switch_vertex(*this);
    size_t vid = vert_tuple.vid(*this);

    if (m_params.do_projection)
    {
        Eigen::VectorXd sqrD;
        int fid;
        Eigen::RowVector3d C;
        tree.squared_distance(input_V, input_F, Eigen::RowVector3d(V), fid, C);
        V = C;
    }

    auto& v = vertex_attrs[vid];
    v.pos = uv;
    v.pos_3d = V;

    Eigen::MatrixXd V_local, uv_local, Ji;
    Eigen::MatrixXi F_local;
    Eigen::SparseMatrix<double> G_local;
    get_mesh_onering(t.switch_vertex(*this), V_local, uv_local, F_local);
    Eigen::VectorXd area, area_3d;
    igl::doublearea(V_local, F_local, area_3d);
    igl::doublearea(uv_local, F_local, area);

    if (area.minCoeff() <= 0 || area_3d.minCoeff() <= 0)
    {
        return false;
    }
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    auto Es = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));

    for (int i = 0; i < Es.rows(); i++)
    {
        if (!std::isfinite(Es(i)) || std::isnan(Es(i)))
        {
            return false;
        }
    }

    // if use projection, we need to check E_max
    if (m_params.do_projection)
    {
        if (Es.maxCoeff() > position_cache.local().E_max_before_collpase)
        {
            return false;
        }
    }
        
    for (size_t nbr_vid : get_one_ring_vids_for_vertex(vid)) {
        if (nbr_vid != vid && vertex_attrs[nbr_vid].pos == v.pos) {
            return false;
        }
    }


    return true;
}
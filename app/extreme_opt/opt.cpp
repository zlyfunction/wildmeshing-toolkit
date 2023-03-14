
#include "ExtremeOpt.h"
#include "wmtk/ExecutionScheduler.hpp"

#include <Eigen/src/Core/util/Constants.h>
#include <igl/Timer.h>
#include <igl/cat.h>
#include <igl/grad.h>
#include <igl/local_basis.h>
#include <wmtk/utils/AMIPS2D.h>
#include <Eigen/Sparse>
#include <array>
#include <wmtk/utils/Logger.hpp>
#include <wmtk/utils/TriQualityUtils.hpp>

#include <igl/boundary_loop.h>
#include <igl/predicates/predicates.h>
#include <igl/upsample.h>
#include <igl/writeOBJ.h>
#include <limits>
#include <optional>
#include <wmtk/utils/TupleUtils.hpp>
#include "SYMDIR.h"
#include "rref.h"

void buildAeq(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& Aeq)
{
    int N = uv.rows();
    int c = 0;
    int m = EE.rows() / 2;

    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);

    int n_fix_dof = 3;

    std::set<std::pair<int, int>> added_e;

    Aeq.resize(2 * m + n_fix_dof, uv.rows() * 2);
    int A2, B2, C2, D2;
    for (int i = 0; i < EE.rows(); i++) {
        int A2 = EE(i, 0);
        int B2 = EE(i, 1);
        int C2 = EE(i, 2);
        int D2 = EE(i, 3);
        auto e0 = std::make_pair(A2, B2);
        auto e1 = std::make_pair(C2, D2);
        if (added_e.find(e0) != added_e.end() || added_e.find(e1) != added_e.end()) continue;
        added_e.insert(e0);
        added_e.insert(e1);

        Eigen::Matrix<double, 2, 1> e_ab = uv.row(B2) - uv.row(A2);
        Eigen::Matrix<double, 2, 1> e_dc = uv.row(C2) - uv.row(D2);

        Eigen::Matrix<double, 2, 1> e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        int r = (int)(round(2 * angle / double(igl::PI)) + 2) % 4;

        std::vector<Eigen::Matrix<double, 2, 2>> r_mat(4);
        r_mat[0] << -1, 0, 0, -1;
        r_mat[1] << 0, 1, -1, 0;
        r_mat[2] << 1, 0, 0, 1;
        r_mat[3] << 0, -1, 1, 0;

        Aeq.coeffRef(c, A2) += 1;
        Aeq.coeffRef(c, B2) += -1;
        Aeq.coeffRef(c + 1, A2 + N) += 1;
        Aeq.coeffRef(c + 1, B2 + N) += -1;

        Aeq.coeffRef(c, C2) += r_mat[r](0, 0);
        Aeq.coeffRef(c, D2) += -r_mat[r](0, 0);
        Aeq.coeffRef(c, C2 + N) += r_mat[r](0, 1);
        Aeq.coeffRef(c, D2 + N) += -r_mat[r](0, 1);
        Aeq.coeffRef(c + 1, C2) += r_mat[r](1, 0);
        Aeq.coeffRef(c + 1, D2) += -r_mat[r](1, 0);
        Aeq.coeffRef(c + 1, C2 + N) += r_mat[r](1, 1);
        Aeq.coeffRef(c + 1, D2 + N) += -r_mat[r](1, 1);
        c = c + 2;
    }

    double min_u_diff = 1e10;
    int min_u_diff_id = 0;
    auto l = bds[0];
    for (int i = 0; i < l.size(); i++) {
        double u_diff = abs(uv(l[i], 0) - uv(l[(i + 1) % l.size()], 0));
        if (u_diff < min_u_diff) {
            min_u_diff = u_diff;
            min_u_diff_id = i;
        }
    }

    std::cout << "fix " << l[min_u_diff_id] << std::endl;
    Aeq.coeffRef(c, l[min_u_diff_id]) = 1;
    Aeq.coeffRef(c + 1, l[min_u_diff_id] + N) = 1;
    c = c + 2;
    std::cout << "fix " << l[(min_u_diff_id + 1) % l.size()] << std::endl;
    Aeq.coeffRef(c, l[(min_u_diff_id + 1) % l.size()]) = 1;
    c = c + 1;
}

void buildkkt(
    Eigen::SparseMatrix<double>& hessian,
    Eigen::SparseMatrix<double>& Aeq,
    Eigen::SparseMatrix<double>& AeqT,
    Eigen::SparseMatrix<double>& kkt)
{
    kkt.reserve(hessian.nonZeros() + Aeq.nonZeros() + AeqT.nonZeros());
    for (Eigen::Index c = 0; c < kkt.cols(); ++c) {
        kkt.startVec(c);
        if (c < hessian.cols()) {
            for (typename Eigen::SparseMatrix<double>::InnerIterator ithessian(hessian, c);
                 ithessian;
                 ++ithessian)
                kkt.insertBack(ithessian.row(), c) = ithessian.value();
            for (typename Eigen::SparseMatrix<double>::InnerIterator itAeq(Aeq, c); itAeq; ++itAeq)
                kkt.insertBack(itAeq.row() + hessian.rows(), c) = itAeq.value();
        } else {
            for (typename Eigen::SparseMatrix<double>::InnerIterator itAeqT(
                     AeqT,
                     c - hessian.cols());
                 itAeqT;
                 ++itAeqT)
                kkt.insertBack(itAeqT.row(), c) = itAeqT.value();
        }
    }
    kkt.finalize();
}

int check_flip(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& Fn)
{
    int fl = 0;
    for (int i = 0; i < Fn.rows(); i++) {
        Eigen::Matrix<double, 1, 2> a_db(uv(Fn(i, 0), 0), uv(Fn(i, 0), 1));
        Eigen::Matrix<double, 1, 2> b_db(uv(Fn(i, 1), 0), uv(Fn(i, 1), 1));
        Eigen::Matrix<double, 1, 2> c_db(uv(Fn(i, 2), 0), uv(Fn(i, 2), 1));
        if (igl::predicates::orient2d(a_db, b_db, c_db) != igl::predicates::Orientation::POSITIVE) {
            fl++;
        }
    }
    return fl;
}

using namespace wmtk;
auto renew = [](auto& m, auto op, auto& tris) {
    auto edges = m.new_edges_after(tris);
    auto optup = std::vector<std::pair<std::string, wmtk::TriMesh::Tuple>>();
    for (auto& e : edges) optup.emplace_back(op, e);
    return optup;
};

auto renew_collapse = [](auto& m, auto op, auto& tris) {
    auto edges = m.new_edges_after(tris);
    auto optup = std::vector<std::pair<std::string, wmtk::TriMesh::Tuple>>();
    for (auto& e : edges) {
        if (m.m_params.with_cons)
        {
            if (m.is_boundary_edge(e)) {
                optup.emplace_back("test_op", e);
            } else {
                optup.emplace_back("edge_collapse", e);
            }
        }
        else
        {
            optup.emplace_back("edge_collapse", e);
        }
        
    }
    return optup;
};


namespace extremeopt {
void get_grad_op(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& grad_op)
{
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);

    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G, false);

    auto face_proj = [](Eigen::MatrixXd& F) {
        std::vector<Eigen::Triplet<double>> IJV;
        int f_num = F.rows();
        for (int i = 0; i < F.rows(); i++) {
            IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
            IJV.push_back(Eigen::Triplet<double>(i, i + f_num, F(i, 1)));
            IJV.push_back(Eigen::Triplet<double>(i, i + 2 * f_num, F(i, 2)));
        }
        Eigen::SparseMatrix<double> P(f_num, 3 * f_num);
        P.setFromTriplets(IJV.begin(), IJV.end());
        return P;
    };

    Eigen::SparseMatrix<double> Dx = face_proj(F1) * G;
    Eigen::SparseMatrix<double> Dy = face_proj(F2) * G;

    Eigen::SparseMatrix<double> hstack = igl::cat(1, Dx, Dy);
    Eigen::SparseMatrix<double> empty(hstack.rows(), hstack.cols());

    grad_op = igl::cat(1, igl::cat(2, hstack, empty), igl::cat(2, empty, hstack));
}
} // namespace extremeopt

double extremeopt::ExtremeOpt::get_e_max_onering(const Tuple& t)
{
    Eigen::MatrixXd V_local, uv_local, Ji;
    Eigen::MatrixXi F_local;
    Eigen::SparseMatrix<double> G_local;
    get_mesh_onering(t, V_local, uv_local, F_local);
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    return wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();
}

void extremeopt::ExtremeOpt::cache_edge_positions(const Tuple& t)
{
    position_cache.local().V1 = vertex_attrs[t.vid(*this)].pos_3d;
    position_cache.local().V2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos_3d;
    position_cache.local().uv1 = vertex_attrs[t.vid(*this)].pos;
    position_cache.local().uv2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos;

    position_cache.local().vid1 = t.vid(*this);
    position_cache.local().vid2 = t.switch_vertex(*this).vid(*this);

    position_cache.local().is_v1_bd = this->is_boundary_vertex(t);
    position_cache.local().is_v2_bd = this->is_boundary_vertex(t.switch_vertex(*this));

    if (position_cache.local().is_v1_bd) {
        auto onering_e = get_one_ring_edges_for_vertex(t);
        int cnt = 0;
        for (auto e : onering_e) {
            if (is_boundary_edge(e)) {
                if (cnt == 0) {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e1 = e;
                    else
                        position_cache.local().bd_e1 = e.switch_vertex(*this);
                } else {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e2 = e;
                    else
                        position_cache.local().bd_e2 = e.switch_vertex(*this);
                }
                cnt++;
            }
            if (cnt == 2) break;
        }
    } else if (position_cache.local().is_v2_bd) {
        auto onering_e = get_one_ring_edges_for_vertex(t.switch_vertex(*this));
        int cnt = 0;
        for (auto e : onering_e) {
            if (is_boundary_edge(e)) {
                if (cnt == 0) {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e1 = e;
                    else
                        position_cache.local().bd_e1 = e.switch_vertex(*this);
                } else {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e2 = e;
                    else
                        position_cache.local().bd_e2 = e.switch_vertex(*this);
                }
                cnt++;
            }
            if (cnt == 2) break;
        }
    }
    double E1, E2;
    E1 = get_e_max_onering(t);
    E2 = get_e_max_onering(t.switch_vertex(*this));

    position_cache.local().E_max_before_collpase = std::max(E1, E2);
}

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
    // if((v.pos.array() != 0).all() ||
    //(v.pos_3d.array() != 0).all()) {
    //     spdlog::error("writing to a nontrivial vertex after a split, should be a new one! {} ::
    //     {}",
    //             fmt::join(v.pos,","),
    //             fmt::join(v.pos_3d,",")
    //             );
    // }
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

void extremeopt::ExtremeOpt::split_all_edges(const Eigen::VectorXd &Es)
{
    Eigen::MatrixXi EE;
    if (m_params.with_cons)
    {
        export_EE(EE);
    }
    size_t vid_threshold = 0;
    auto collect_all_ops_split = std::vector<std::pair<std::string, Tuple>>();

    for (auto& loc : get_faces())
    {
        auto t0 = loc;
        auto t1 = t0.switch_edge(*this);
        auto t2 = t0.switch_vertex(*this).switch_edge(*this);
        double l0 =
            (vertex_attrs[t0.vid(*this)].pos - vertex_attrs[t0.switch_vertex(*this).vid(*this)].pos)
                .norm();
        double l1 =
            (vertex_attrs[t1.vid(*this)].pos - vertex_attrs[t1.switch_vertex(*this).vid(*this)].pos)
                .norm();
        double l2 =
            (vertex_attrs[t2.vid(*this)].pos - vertex_attrs[t2.switch_vertex(*this).vid(*this)].pos)
                .norm();
        if (is_boundary_edge(t0)) l0 = 0;
        if (is_boundary_edge(t1)) l1 = 0;
        if (is_boundary_edge(t2)) l2 = 0;

        if (l0 >= l1 && l0 >= l2) {
            collect_all_ops_split.emplace_back("edge_split", t0);
        } else if (l1 >= l2) {
            collect_all_ops_split.emplace_back("edge_split", t1);

        } else {
            collect_all_ops_split.emplace_back("edge_split", t2);
        }
    }
    
    
    auto setup_and_execute = [&](auto& executor_split) {
        executor_split.priority = [&](auto& m, auto _, auto& e) {
            if (e.fid(*this) >= Es.size()) return 1e50;
            else return Es(e.fid(*this));
        };
        executor_split.stopping_criterion_checking_frequency = 300;
        executor_split.stopping_criterion = [](const TriMesh&) {
            return true; // non-stop, process everything
        };
        executor_split.num_threads = NUM_THREADS;
        executor_split(*this, collect_all_ops_split);
    };
    auto executor_split = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    setup_and_execute(executor_split);
    
    if (m_params.with_cons)
    {
        update_constraints_EE_v(EE);
    }
}

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

    double E_max_after_collapse;
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    auto E_max_after_collapses =
        wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    
    for (int i = 0; i < E_max_after_collapses.size(); i++)
    {
        if (!std::isfinite(E_max_after_collapses(i)))
        {
            return false;
        }
    }
    E_max_after_collapse = E_max_after_collapses.maxCoeff();
    if (E_max_after_collapse > position_cache.local().E_max_before_collpase) {
        // E_max does not go down
        return false;
    }

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
    double& E_max)
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
    E_max = Es.maxCoeff();

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

std::vector<wmtk::TriMesh::Tuple> extremeopt::ExtremeOpt::new_edges_after(
    const std::vector<wmtk::TriMesh::Tuple>& tris) const
{
    std::vector<wmtk::TriMesh::Tuple> new_edges;

    for (auto t : tris) {
        for (auto j = 0; j < 3; j++) {
            new_edges.push_back(wmtk::TriMesh::tuple_from_edge(t.fid(*this), j));
        }
    }
    wmtk::unique_edge_tuples(*this, new_edges);

    return new_edges;
}

std::vector<wmtk::TriMesh::Tuple> extremeopt::ExtremeOpt::replace_edges_after_split(
    const std::vector<wmtk::TriMesh::Tuple>& tris,
    const size_t vid_threshold) const
{
    std::vector<wmtk::TriMesh::Tuple> new_edges;

    for (auto t : tris) {
        auto tmptup = (t.switch_vertex(*this)).switch_edge(*this);
        if (tmptup.vid(*this) < vid_threshold &&
            (tmptup.switch_vertex(*this)).vid(*this) < vid_threshold)
            new_edges.push_back(tmptup);
    }
    wmtk::unique_edge_tuples(*this, new_edges);
    return new_edges;
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

bool extremeopt::ExtremeOpt::smooth_before(const Tuple& t)
{
    if (!t.is_valid(*this)) {
        std::cout << "tuple not valid" << std::endl;
        return false;
    }

    // if (is_boundary_vertex(t)) {
    //     return false;
    // }
    // // it's okay to move the boundary(for now)
    // if (vertex_attrs[t.vid(*this)].fixed)
    //     return false;
    return true;
}


bool extremeopt::ExtremeOpt::smooth_after(const Tuple& t)
{
    // Newton iterations are encapsulated here.
    wmtk::logger().trace("Newton iteration for vertex smoothing.");
    auto vid = t.vid(*this);

    wmtk::logger().trace("smooth vertex {}", vid);
    // std::cout << "vertex " << vid << std::endl;
    bool ls_good = false;

    if (!is_boundary_vertex(t))
    { 
        return false; // TODO: comment out this
        auto vid_onering = get_one_ring_vids_for_vertex(vid);
        auto locs = get_one_ring_tris_for_vertex(t);
        assert(locs.size() > 0);

        // get local (V, F)
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
        Eigen::VectorXd area_local(locs.size());
        for (size_t i = 0; i < locs.size(); i++) {
            int t_id = locs[i].fid(*this);
            auto local_tuples = oriented_tri_vertices(locs[i]);
            for (size_t j = 0; j < 3; j++) {
                F_local(i, j) = v_map[local_tuples[j].vid(*this)];
            }
            // area_local(i) = face_attrs[t_id].area_3d;
        }
        igl::doublearea(V_local, F_local, area_local);
        Eigen::SparseMatrix<double> G_local;
        get_grad_op(V_local, F_local, G_local);
        // std::cout << "G_local: \n" << G_local << std::endl;

        auto compute_energy = [&G_local, &area_local](Eigen::MatrixXd& aaa) {
            Eigen::MatrixXd Ji;
            wmtk::jacobian_from_uv(G_local, aaa, Ji);
            return wmtk::compute_energy_from_jacobian(Ji, area_local);
        };


        Eigen::SparseMatrix<double> hessian_local;
        Eigen::VectorXd grad_local;

        double local_energy_0 = wmtk::get_grad_and_hessian(
            G_local,
            area_local,
            uv_local,
            grad_local,
            hessian_local,
            m_params.do_newton);
        Eigen::MatrixXd search_dir(1, 2);
        if (!m_params.do_newton) {
            search_dir =
                -Eigen::Map<Eigen::MatrixXd>(grad_local.data(), uv_local.rows(), 2).row(v_map[vid]);
        } else {
            // local hessian for only one node
            Eigen::Matrix2d hessian_at_v;
            hessian_at_v << hessian_local.coeff(v_map[vid], v_map[vid]),
                hessian_local.coeff(v_map[vid], v_map[vid] + vid_onering.size()),
                hessian_local.coeff(v_map[vid] + vid_onering.size(), v_map[vid]),
                hessian_local.coeff(v_map[vid] + vid_onering.size(), v_map[vid] + vid_onering.size());
            Eigen::Vector2d grad_at_v;
            grad_at_v << grad_local(v_map[vid]), grad_local(v_map[vid] + vid_onering.size());
            Eigen::Vector2d newton_at_v = hessian_at_v.ldlt().solve(-grad_at_v);
            search_dir << newton_at_v(0), newton_at_v(1);
        }
        // std::cout << "search_dir" << search_dir << std::endl;
        // do linesearch
        // std::cout << "local E0 = " << local_energy_0 << std::endl;
        auto pos_copy = vertex_attrs[vid].pos;
        double step = 1.0;
        double new_energy;
        auto new_x = uv_local;
        for (int i = 0; i < m_params.ls_iters; i++) {
            new_x.row(v_map[vid]) = uv_local.row(v_map[vid]) + step * search_dir;
            vertex_attrs[vid].pos << new_x(v_map[vid], 0), new_x(v_map[vid], 1);
            new_energy = compute_energy(new_x);
            // std::cout << "new E " << new_energy << std::endl;

            bool has_flip = false;
            for (auto loc : locs) {
                if (is_inverted(loc)) {
                    has_flip = true;
                    break;
                }
            }
            if (new_energy < local_energy_0 && !has_flip) {
                ls_good = true;
                break;
            }
            step = step * 0.8;
        }
        if (ls_good) {
            wmtk::logger()
                .trace("ls good, step = {}, energy {} -> {}", step, local_energy_0, new_energy);
        } else {
            wmtk::logger().trace("ls failed");
            vertex_attrs[vid].pos = pos_copy;
        }
    }
    else // boudnary vertex
    {
        std::vector<Tuple> ts;
        std::vector<Eigen::MatrixXd> Vs, uvs;
        std::vector<Eigen::MatrixXi> Fs;
        std::vector<Eigen::Matrix2d> rots;
        std::vector<Eigen::VectorXd> areas;
        std::vector<Eigen::SparseMatrix<double>> Gs;
        std::vector<Eigen::Matrix<double, 2, 2>> r_mat(4);
        r_mat[0] << -1, 0, 0, -1;
        r_mat[1] << 0, 1, -1, 0;
        r_mat[2] << 1, 0, 0, 1;
        r_mat[3] << 0, -1, 1, 0;
        std::vector<int> local_vids;
        bool flag = true, is_first = true;
        auto t_cur = t;
        while (flag)
        {
            // std::cout << "t_cur_now" << std::endl;
            // t_cur.print_infos();

            ts.push_back(t_cur);
            Eigen::MatrixXd V_local, uv_local;
            Eigen::MatrixXi F_local;
            local_vids.push_back(get_mesh_onering(t_cur, V_local, uv_local, F_local));
            Vs.push_back(V_local); uvs.push_back(uv_local); Fs.push_back(F_local);
            Eigen::SparseMatrix<double> G_local;
            get_grad_op(V_local, F_local, G_local);
            Gs.push_back(G_local);
            Eigen::VectorXd area_local;
            igl::doublearea(V_local, F_local, area_local);
            areas.push_back(area_local);
            auto ve = get_one_ring_edges_for_vertex(t_cur);
            Tuple local_bd;
            for (auto e : ve)
            {
                if (is_boundary_edge(e) && (is_first || e.eid(*this) != t_cur.eid(*this))) local_bd = e;
            }
            // std::cout << "local_bd_0" << std::endl;
            // local_bd.print_info();
            is_first = false;
            bool do_switch = false;
            if (!local_bd.is_ccw(*this)) 
            {
                local_bd = local_bd.switch_vertex(*this);
                do_switch = true;
            }
            // std::cout << "local_bd_1" << std::endl;
            // local_bd.print_info();
            t_cur = edge_attrs[local_bd.eid(*this)].pair;
            // std::cout << "computed t_cur" << std::endl;
            // t_cur.print_info();
        

            if (t_cur.vid(*this) == t.vid(*this) || t_cur.switch_vertex(*this).vid(*this) == t.vid(*this))
            {
                flag = false;
            }
            if (flag)
            {
                Eigen::Vector2d e_ab = vertex_attrs[t_cur.switch_vertex(*this).vid(*this)].pos - vertex_attrs[t_cur.vid(*this)].pos;
                Eigen::Vector2d e_dc = vertex_attrs[local_bd.vid(*this)].pos - vertex_attrs[local_bd.switch_vertex(*this).vid(*this)].pos;

                Eigen::Vector2d e_ab_perp;
                e_ab_perp(0) = -e_ab(1);
                e_ab_perp(1) = e_ab(0);
                double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
                int r = (int)(round(2 * angle / igl::PI) + 2) % 4;

                rots.push_back(r_mat[r]);

                // std::cout << "check rotation" << std::endl;
                // std::cout << "R*ab:\n" << r_mat[r] * e_ab << std::endl;
                // std::cout << "cd:" << -e_dc << std::endl; 
            }
            if (do_switch) t_cur = t_cur.switch_vertex(*this);
        } // end of while loop
        double total_area = 0.0;
        for (int i = 0; i < areas.size(); i++)
        {
            total_area += areas[i].sum();
        }
        std::vector<Eigen::Matrix<double, 1, 2>> grads(Vs.size());
        std::vector<double> E0s;
        for (int i = 0; i < Vs.size(); i++)
        {
            Eigen::SparseMatrix<double> hessian_local;
            Eigen::VectorXd grad_local;
            double local_energy_0 = wmtk::get_grad_and_hessian(
                Gs[i],
                areas[i],
                uvs[i],
                grad_local,
                hessian_local,
                false);
            grads[i] = -Eigen::Map<Eigen::MatrixXd>(grad_local.data(), uvs[i].rows(), 2).row(local_vids[i]);
        }
        
        std::vector<Eigen::Matrix<double, 1, 2>> dirs(Vs.size());
        Eigen::Matrix<double, 1, 2> dir = areas[0].sum() * grads[0];
        for (int i = 0; i < Vs.size() - 1; i++)
        {
            Eigen::Matrix<double, 1, 2> local_dir = grads[i + 1];
            for (int j = i; j >= 0; j--)
            {
                local_dir = (rots[j].transpose() * dir.transpose()).transpose();
            }
            dir += areas[i].sum() * local_dir;
        }
        dirs[0] = dir / total_area;
        for (int i = 0; i < Vs.size() - 1; i++)
        {
            dirs[i + 1] = (rots[i] * dirs[i].transpose()).transpose();
        }

        
        // std::cout << "information for smoothing vertex: " << t.vid(*this) << std::endl;
        // std::cout << "copies " << ts.size() << std::endl;
        // for (int i = 0; i < ts.size(); i++)
        // {
        //     std::cout << ts[i].vid(*this) << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "check their 3d position" << std::endl;
        // for (int i = 0; i < ts.size(); i++)
        // {
        //     std::cout << vertex_attrs[ts[i].vid(*this)].pos_3d << std::endl;
        // }
        // std::cout << "dirs" << std::endl;
        // for (int i = 0; i < dirs.size(); i++)
        // {
        //     std::cout << dirs[i] << std::endl;
        // }
        // std::cout << std::endl;
    }

    
    return ls_good;
}

void extremeopt::ExtremeOpt::smooth_global(int steps)
{
    Eigen::MatrixXi F;
    Eigen::MatrixXd V, uv;
    export_mesh(V, F, uv);
    Eigen::MatrixXi EE;
    export_EE(EE);

    Eigen::VectorXd area;
    Eigen::SparseMatrix<double> G;
    igl::doublearea(V, F, area);
    get_grad_op(V, F, G);
    Eigen::SparseMatrix<double> Aeq;
    buildAeq(EE, uv, F, Aeq);
    Eigen::SparseMatrix<double> AeqT = Aeq.transpose();

    auto compute_energy = [G, area](Eigen::MatrixXd aaa) {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G, aaa, Ji);
        return wmtk::compute_energy_from_jacobian(Ji, area);
    };
    Eigen::VectorXd newton;
    // get grad and hessian
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd grad;
    double energy_0 = wmtk::get_grad_and_hessian(G, area, uv, grad, hessian, m_params.do_newton);
    
    bool use_rref = true;
    if (!use_rref)
    {
        // build kkt system
        Eigen::SparseMatrix<double> kkt(hessian.rows() + Aeq.rows(), hessian.cols() + Aeq.rows());
        buildkkt(hessian, Aeq, AeqT, kkt);
        Eigen::VectorXd rhs(kkt.rows());
        rhs.setZero();
        rhs.topRows(grad.rows()) << -grad;
        // solve the system
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(kkt);
        solver.factorize(kkt);
        newton = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            std::cout << "cannot solve newton system" << std::endl;
            hessian.setIdentity();
            buildkkt(hessian, Aeq, AeqT, kkt);
            solver.analyzePattern(kkt);
            solver.factorize(kkt);
            newton = solver.solve(rhs);
        }
    }
    else
    {
        Eigen::SparseMatrix<double> Q2(Aeq.cols(), Aeq.cols() - Aeq.rows()), Q2T;
        elim_constr(Aeq, Q2);
        Q2.makeCompressed();
        Q2T = Q2.transpose();
        // std::cout << "test q2:" << (Aeq * Q2 * Eigen::VectorXd::Random(Q2.cols())).norm() << std::endl;
        hessian = Q2T * hessian * Q2;
        grad = Q2T * grad;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(hessian);
        solver.factorize(hessian);
        newton = solver.solve(-grad);
        if (solver.info() != Eigen::Success) {
            std::cout << "cannot solve newton system" << std::endl;
            newton = -grad;
        }
        newton = Q2 * newton;
    }

    // do lineserach
    Eigen::MatrixXd search_dir = Eigen::Map<Eigen::MatrixXd>(newton.data(), V.rows(), 2);
    auto new_x = uv;
    double ls_step_size = 1.0;
    bool ls_good = false;
    for (int i = 0; i < m_params.ls_iters; i++) {
        new_x = uv + ls_step_size * search_dir;
        double new_E = compute_energy(new_x);
        if (new_E < energy_0 && check_flip(new_x, F) == 0) {
            ls_good = true;
            break;
        }
        ls_step_size *= 0.8;
    }

    if (ls_good) {
        // update vertex_attrs
        std::cout << "ls_step_size = " << ls_step_size << std::endl;
        for (int i = 0; i < new_x.rows(); i++) {
            vertex_attrs[i].pos = new_x.row(i);
        }
    } else {
        std::cout << "smooth failed" << std::endl;
    }
}

// TODO: seperate smooth operations and swap operations as 2 functions
void extremeopt::ExtremeOpt::smooth_all_vertices()
{
    auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_vertices()) {
        collect_all_ops.emplace_back("vertex_smooth", loc);
    }

    auto executor = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    executor(*this, collect_all_ops);
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

void extremeopt::ExtremeOpt::collapse_all_edges()
{
    auto collect_all_ops_collapse = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_edges()) {
        if (m_params.with_cons)
        {
            if (is_boundary_edge(loc)) {
                collect_all_ops_collapse.emplace_back("test_op", loc);
            } else {
                collect_all_ops_collapse.emplace_back("edge_collapse", loc);
            }
        }
        else
        {
            collect_all_ops_collapse.emplace_back("edge_collapse", loc);
        }
        // collect_all_ops_collapse.emplace_back("test_op", loc);
    }
    auto setup_and_execute = [&](auto& executor_collapse) {
        executor_collapse.renew_neighbor_tuples = renew_collapse;
        // add term with energy
        executor_collapse.priority = [&](auto& m, auto _, auto& e) {
            return (vertex_attrs[e.vid(*this)].pos -
                    vertex_attrs[e.switch_vertex(*this).vid(*this)].pos)
                .norm();
        };
        executor_collapse.num_threads = NUM_THREADS;
        executor_collapse(*this, collect_all_ops_collapse);
        // TODO: priority queue (edge length)
    };
    std::map<Op, std::function<std::optional<std::vector<Tuple>>(ExtremeOpt&, const Tuple&)>>
        test_op = {
            {"test_op", [](ExtremeOpt& m, const Tuple& t) -> std::optional<std::vector<Tuple>> {
                 //                 auto retdata = CollapsePair()(t, m);
                 //                 if (retdata.success) {
                 //                     return retdata.new_tris;
                 //                 } else {
                 //                     return {};
                 //                 }
                 std::vector<Tuple> ret;

                 ExtremeOpt::CollapsePair ce_op;
                 if (auto [new_t, succ] = ce_op.execute(t, m, ret); succ) {
                     return ret;
                 }
                 return {};
             }}};
    auto executor_collapse = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>(test_op);
    setup_and_execute(executor_collapse);
}


void extremeopt::ExtremeOpt::do_optimization(json& opt_log)
{
    igl::Timer timer;
    double time;

    // prepare to compute energy
    Eigen::SparseMatrix<double> G_global;
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;

    std::cout << "before export" << std::endl;
    export_mesh(V, F, uv);


    // get edge length thresholds for collapsing operation
    elen_threshold = 0;
    elen_threshold_3d = 0;
    for (int i = 0; i < F.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int v1 = F(i, j);
            int v2 = F(i, (j+1)%3);
            double elen = (uv.row(v1) - uv.row(v2)).norm();
            double elen_3d = (V.row(v1) - V.row(v2)).norm();
            if (elen > elen_threshold) elen_threshold = elen;
            if (elen_3d > elen_threshold_3d) elen_threshold_3d = elen_3d;
        }
    }
    elen_threshold *= m_params.elen_alpha;
    elen_threshold_3d *= m_params.elen_alpha;
    get_grad_op(V, F, G_global);
    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);
    auto compute_energy = [&G_global, &dblarea](Eigen::MatrixXd& aaa) {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_global, aaa, Ji);

        return wmtk::compute_energy_from_jacobian(Ji, dblarea);
    };

    auto compute_energy_max = [&G_global, &dblarea, &F, &V](Eigen::MatrixXd& aaa) {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_global, aaa, Ji);
        auto EVec = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));

        for (int j = 0; j < EVec.size(); ++j) {
            if (!std::isfinite(EVec(j))) {
                auto f = F.row(j);
                spdlog::info(
                    "triangle {} was not finite area {} ({})",
                    j,
                    dblarea(j),
                    fmt::join(f, ","));
            }
        }
       
        return EVec.maxCoeff();
    };
    auto compute_energy_all = [&G_global, &dblarea, &F](Eigen::MatrixXd& aaa) {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_global, aaa, Ji);
        return wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    };

    double E = compute_energy(uv);
    wmtk::logger().info("Start Energy E = {}", E);
    opt_log["opt_log"].push_back(
                {{"F_size", F.rows()}, {"V_size", V.rows()}, {"E_max", compute_energy_max(uv)}, {"E_avg", E}});
    double E_old = E;
    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_max;

        if (this->m_params.do_split) {
            auto Es = compute_energy_all(uv);
            split_all_edges(Es);

            export_mesh(V, F, uv);
            get_grad_op(V, F, G_global);
            igl::doublearea(V, F, dblarea);
            E = compute_energy(uv);
            wmtk::logger()
                .info("Mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());
            wmtk::logger().info("After splitting, E = {}", E);
            wmtk::logger().info("E_max = {}", compute_energy_max(uv));
            spdlog::info("E is {} {} {}", std::isfinite(E), !std::isnan(E), !std::isinf(E));
        }
        // igl::writeOBJ("new_tests/step_" + std::to_string(i) + "_split.obj", V, F, V, F, uv, F);

        if (this->m_params.local_smooth) {
            smooth_all_vertices();
            export_mesh(V, F, uv);
            get_grad_op(V, F, G_global);
            igl::doublearea(V, F, dblarea);
            E = compute_energy(uv);
            E_max = compute_energy_max(uv);
            wmtk::logger().info("After LOCAL smoothing {}, E = {}", i, E);
            wmtk::logger().info("E_max = {}", E_max);
        }

        if (this->m_params.do_swap) {
            timer.start();
            swap_all_edges();
            time = timer.getElapsedTime();
            wmtk::logger().info("edges swapping operation time serial: {}s", time);
            export_mesh(V, F, uv);

            get_grad_op(V, F, G_global);
            igl::doublearea(V, F, dblarea);
            E = compute_energy(uv);
            E_max = compute_energy_max(uv);
            wmtk::logger().info("After swapping, E = {}", E);
            wmtk::logger().info("E_max = {}", E_max);
        }
        // igl::writeOBJ("new_tests/step_" + std::to_string(i) + "_swap.obj", V, F, V, F, uv, F);

        if (this->m_params.do_collapse) {
            collapse_all_edges();
            export_mesh(V, F, uv);
            get_grad_op(V, F, G_global);
            igl::doublearea(V, F, dblarea);
            E = compute_energy(uv);
            wmtk::logger()
                .info("Mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());
            wmtk::logger().info("After collapsing, E = {}", E);
            E_max = compute_energy_max(uv);
            wmtk::logger().info("E_max = {}", E_max);
        }
        // igl::writeOBJ("new_tests/step_" + std::to_string(i) + "_collapse.obj", V, F, V, F, uv, F);

        if (this->m_params.global_smooth) {
            timer.start();
            smooth_global(1);
            time = timer.getElapsedTime();
            wmtk::logger().info("GLOBAL smoothing operation time serial: {}s", time);
            export_mesh(V, F, uv);
            get_grad_op(V, F, G_global);
            igl::doublearea(V, F, dblarea);
            E = compute_energy(uv);
            E_max = compute_energy_max(uv);
            wmtk::logger().info("After GLOBAL smoothing {}, E = {}", i, E);
            wmtk::logger().info("E_max = {}", E_max);

            opt_log["opt_log"].push_back(
                {{"F_size", F.rows()}, {"V_size", V.rows()}, {"E_max", E_max}, {"E_avg", E}});
        }
        // igl::writeOBJ("new_tests/step_" + std::to_string(i) + "_smooth.obj", V, F, V, F, uv, F);

        // terminate criteria
        // if (E < m_params.E_target) {
        //     wmtk::logger().info(
        //         "Reach target energy({}), optimization succeed!",
        //         m_params.E_target);
        //     break;
        // }
        // if (E == E_old) {
        //     wmtk::logger().info("Energy get stuck, optimization failed.");
        //     break;
        // }

        E_old = E;
        std::cout << std::endl;
    }
}

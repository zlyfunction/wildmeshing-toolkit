#include "ExtremeOpt.h"
#include <wmtk/ExecutionScheduler.hpp>
#include "SYMDIR.h"
#include "rref.h"
#include <igl/predicates/predicates.h>
#include <igl/boundary_loop.h>
#include <igl/writeOBJ.h>
#include "SYMDIR_NEW.h"

namespace {

using namespace extremeopt;
using namespace wmtk;

class ExtremeOptSmoothVertexOperation : public wmtk::TriMeshOperationShim<
                                                  ExtremeOpt,
                                                  ExtremeOptSmoothVertexOperation,
                                                  wmtk::TriMeshSmoothVertexOperation>
{
public:
    ExecuteReturnData execute(ExtremeOpt& m, const Tuple& t)
    {
        return wmtk::TriMeshSmoothVertexOperation::execute(m, t);
    }
    bool before(ExtremeOpt& m, const Tuple& t)
    {
        if (wmtk::TriMeshSmoothVertexOperation::before(m, t)) {
            return  m.smooth_before(t);
        }
        return false;
    }
    bool after(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= wmtk::TriMeshSmoothVertexOperation::after(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.smooth_after(ret_data.tuple);
        }
        return ret_data;
    }
    bool invariants(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= wmtk::TriMeshSmoothVertexOperation::invariants(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.invariants(ret_data.new_tris);
        }
        return ret_data;
    }
};

    template <typename Executor>
    void addCustomOps(Executor& e) {

        e.add_operation(std::make_shared<ExtremeOptSmoothVertexOperation>());
    }
}
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

bool extremeopt::ExtremeOpt::smooth_before(const Tuple& t)
{
    if (!t.is_valid(*this)) {
        std::cout << "tuple not valid" << std::endl;
        return false;
    }
    return true;
}

bool extremeopt::ExtremeOpt::smooth_after(const Tuple& t)
{
    // Newton iterations are encapsulated here.
    wmtk::logger().trace("Newton iteration for vertex smoothing.");
    auto vid = t.vid(*this);

    wmtk::logger().trace("smooth vertex {}", vid);
    bool ls_good = false;

    if (!is_boundary_vertex(t) || !m_params.with_cons)
    { 
        // return false;
        auto vid_onering = get_one_ring_vids_for_vertex(vid);
        auto locs = get_one_ring_tris_for_vertex(t);
        assert(locs.size() > 0);


        Eigen::Matrix2d hessian_at_v;
        Eigen::Vector2d grad_at_v;

        wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
        double local_energy_0 = E_eval.get_grad_and_hessian_onering(*this, t, grad_at_v, hessian_at_v);

        Eigen::MatrixXd search_dir(1, 2);
        if (!m_params.do_newton) {
            search_dir << -grad_at_v(0), -grad_at_v(1);
        } else {
            Eigen::Vector2d newton_at_v = hessian_at_v.ldlt().solve(-grad_at_v);
            search_dir << newton_at_v(0), newton_at_v(1);
        }

        auto pos_copy = vertex_attrs[vid].pos;
        double step = 1.0;
        double new_energy;
        for (int i = 0; i < m_params.ls_iters; i++) {
            vertex_attrs[vid].pos << pos_copy + step * search_dir;
            
            bool has_flip = false;
            for (auto loc : locs) {
                if (is_inverted(loc)) {
                    has_flip = true;
                    break;
                }
            }
            new_energy = E_eval.symmetric_dirichlet_energy_onering(*this, t);
            if (std::isfinite(new_energy) && new_energy < local_energy_0 && !has_flip) {
                ls_good = true;
                break;
            }
            else
            {
                // std::cout << "new energy: " << new_energy << ">=" << local_energy_0 << std::endl;
            }
            step = step * 0.8;
        }
        if (ls_good) {
            // wmtk::logger()
            //     .info("vertex {} onering_size {} ls good, step = {}, energy {} -> {}", vid, locs.size(), step, local_energy_0, new_energy);
        } else {
            // wmtk::logger().info("vertex {} ls failed", vid);
            vertex_attrs[vid].pos = pos_copy;
        }
    }
    else if (m_params.with_cons)// boudnary vertex
    {
        return false;
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

            // // TODO: for debug
            // for (int i = 0; i < area_local.rows(); i++)
            // {
            //     area_local(i) = 1.0;
            // }

            areas.push_back(area_local);
            auto ve = get_one_ring_edges_for_vertex(t_cur);
            Tuple local_bd;
            for (auto e : ve)
            {
                if (is_boundary_edge(e) && (is_first || e.eid(*this) != t_cur.eid(*this))) local_bd = e;
            }

            is_first = false;
            bool do_switch = false;
            if (!local_bd.is_ccw(*this)) 
            {
                local_bd = local_bd.switch_vertex(*this);
                do_switch = true;
            }
            t_cur = edge_attrs[local_bd.eid(*this)].pair;
            if (t_cur.vid(*this) == t.vid(*this) || t_cur.switch_vertex(*this).vid(*this) == t.vid(*this))
            {
                flag = false;
            }

            // compute rotation matrix
            Eigen::Vector2d e_ab = vertex_attrs[t_cur.switch_vertex(*this).vid(*this)].pos - vertex_attrs[t_cur.vid(*this)].pos;
            Eigen::Vector2d e_dc = vertex_attrs[local_bd.vid(*this)].pos - vertex_attrs[local_bd.switch_vertex(*this).vid(*this)].pos;
            Eigen::Vector2d e_ab_perp;
            e_ab_perp(0) = -e_ab(1);
            e_ab_perp(1) = e_ab(0);
            double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
            int r = (int)(round(2 * angle / igl::PI) + 2) % 4;
            rots.push_back(r_mat[r]);

            if (do_switch) t_cur = t_cur.switch_vertex(*this);
        } // end of while loop

        if (ts.size() == 1)
        {
            wmtk::logger().info("boundary vertex {} is singularity", t.vid(*this));
            return false;
        }
        Eigen::Matrix2d tmp;
        tmp.setIdentity();
        for (int i = 0; i < rots.size(); i++)
        {
            tmp = tmp * rots[i];
        }
        if (!tmp.isIdentity(1e-10))
        {
            wmtk::logger().info("boundary vertex {} is singularity", t.vid(*this));
            return false;
        }
        
        double total_area = 0.0;
        for (int i = 0; i < areas.size(); i++)
        {
            total_area += areas[i].sum();
        }
        std::vector<Eigen::Matrix<double, 1, 2>> grads(Vs.size());
        double total_energy0 = 0.0;
        double max_energy0 = 0.0;
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
            total_energy0 += areas[i].sum() * local_energy_0;
            
            // duplicate code to get local E_max here
            Eigen::MatrixXd Ji;
            wmtk::jacobian_from_uv(Gs[i], uvs[i], Ji);
            double local_E_max = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();
            max_energy0 = std::max(max_energy0, local_E_max);
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

        auto pos_copy = vertex_attrs[vid].pos;
        double step = 1.0;
        double new_energy = 0.0;
        double new_E_max = 0.0;
        std::vector<Eigen::MatrixXd> new_xs = uvs;
        for (int k = 0; k < m_params.ls_iters; k++)
        {
            new_energy = 0.0;
            new_E_max = 0.0;
            bool has_flip = false;
            for (int i = 0; i < ts.size(); i++)
            {
                new_xs[i].row(local_vids[i]) = uvs[i].row(local_vids[i]) + step * dirs[i];
                vertex_attrs[ts[i].vid(*this)].pos << new_xs[i](local_vids[i], 0), new_xs[i](local_vids[i], 1);
                Eigen::MatrixXd Ji;
                wmtk::jacobian_from_uv(Gs[i], new_xs[i], Ji);
                new_E_max = std::max(new_E_max, wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff());
                new_energy += areas[i].sum() * wmtk::compute_energy_from_jacobian(Ji, areas[i]);
            }
            for (int i = 0; i < ts.size(); i++)
            {
                auto locs = get_one_ring_tris_for_vertex(ts[i]);
                for (auto loc : locs)
                {
                    if (is_inverted(loc))
                    {
                        has_flip = true;
                        break;
                    }
                }
                // check flip here

                if (has_flip)
                {
                    break;
                }
            }
            if (!has_flip && new_energy < total_energy0)
            {
                ls_good = true;
                break;
            }
            step = step * 0.8;
        }
        if (ls_good) {
            int onering_size = 0;
            for (int i = 0; i < Fs.size(); i++)
            {
                onering_size += Fs[i].rows();
            }
            wmtk::logger()
                .info("boundary vertex {}, copies {}, onering_size {}, ls good, step = {}, energy {} -> {}, E_max {} -> {}", t.vid(*this), ts.size(), onering_size, step, total_energy0 / total_area, new_energy / total_area, max_energy0, new_E_max);
        } else {
            wmtk::logger().info("boundary vertex {} ls failed", t.vid(*this));
        }
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
void extremeopt::ExtremeOpt::smooth_all_vertices()
{
    auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_vertices()) {
        collect_all_ops.emplace_back("vertex_smooth", loc);
    }

    auto executor = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    addCustomOps(executor);
    executor(*this, collect_all_ops);
}

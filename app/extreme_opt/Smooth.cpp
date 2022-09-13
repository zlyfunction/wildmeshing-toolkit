
#include "ExtremeOpt.h"
#include "wmtk/ExecutionScheduler.hpp"

#include <Eigen/src/Core/util/Constants.h>
#include <igl/Timer.h>
#include <wmtk/utils/AMIPS2D.h>
#include <array>
#include <wmtk/utils/Logger.hpp>
#include <wmtk/utils/TriQualityUtils.hpp>
#include <Eigen/Sparse>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/cat.h>

#include <limits>
#include <optional>
#include "SYMDIR.h"


namespace extremeopt
{
    void get_grad_op(Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::SparseMatrix<double> &grad_op)
    {
        Eigen::MatrixXd F1, F2, F3;
        igl::local_basis(V, F, F1, F2, F3);
        Eigen::SparseMatrix<double> G;
        igl::grad(V, F, G, false);
        auto face_proj = [](Eigen::MatrixXd &F) {
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
}

bool extremeopt::ExtremeOpt::smooth_before(const Tuple& t)
{
    if (vertex_attrs[t.vid(*this)].fixed)
        return false;
    return true;
}


bool extremeopt::ExtremeOpt::smooth_after(const Tuple& t)
{
    // Newton iterations are encapsulated here.
    wmtk::logger().trace("Newton iteration for vertex smoothing.");
    auto vid = t.vid(*this);
    
    wmtk::logger().trace("smooth vertex {}", vid);
    // std::cout << "vertex " << vid << std::endl;

    auto vid_onering = get_one_ring_vids_for_vertex(vid);
    auto locs = get_one_ring_tris_for_vertex(t);
    assert(locs.size() > 0);
    
    // get local (V, F)
    Eigen::MatrixXd V_local(vid_onering.size(), 3);
    Eigen::MatrixXd uv_local(vid_onering.size(), 2);
    for (size_t i = 0; i < vid_onering.size(); i++)
    {
        V_local.row(i) = vertex_attrs[vid_onering[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[vid_onering[i]].pos;
    }
    std::vector<int> v_map(vertex_attrs.size(), -1);
    for (size_t i = 0; i < vid_onering.size(); i++)
    {
        v_map[vid_onering[i]] = i;
    }
    Eigen::MatrixXi F_local(locs.size(), 3);
    Eigen::VectorXd area_local(locs.size(), 3);
    for (size_t i = 0; i < locs.size(); i++)
    {
        int t_id = locs[i].fid(*this);
        auto local_tuples = oriented_tri_vertices(locs[i]);
        for (size_t j = 0; j < 3; j++)
        {
            F_local(i, j) = v_map[local_tuples[j].vid(*this)];
        }
        area_local(i) = face_attrs[t_id].area_3d;
    }

    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);
    // std::cout << "G_local: \n" << G_local << std::endl;

    auto compute_energy = [&G_local, &area_local](Eigen::MatrixXd &aaa)
    {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_local, aaa, Ji);
        return wmtk::compute_energy_from_jacobian(Ji, area_local);
    };


    Eigen::SparseMatrix<double> hessian_local;
    Eigen::VectorXd grad_local;
    // TODO: set do_newton as param
    bool do_newton = false;

    double local_energy_0 = wmtk::get_grad_and_hessian(G_local, area_local, uv_local, grad_local, hessian_local, do_newton);
    Eigen::MatrixXd search_dir(1, 2);
    if (!do_newton)
    {
        search_dir = -Eigen::Map<Eigen::MatrixXd>(grad_local.data(), uv_local.rows(), 2).row(v_map[vid]);
    }
    else
    {
        // local hessian for only one node
        Eigen::Matrix2d hessian_at_v;
        hessian_at_v << hessian_local.coeff(v_map[vid], v_map[vid]), hessian_local.coeff(v_map[vid], v_map[vid] + vid_onering.size()),
                        hessian_local.coeff(v_map[vid] + vid_onering.size(), v_map[vid]), hessian_local.coeff(v_map[vid] + vid_onering.size(), v_map[vid] + vid_onering.size());
        Eigen::Vector2d grad_at_v;
        grad_at_v << grad_local(v_map[vid]), grad_local(v_map[vid] + vid_onering.size());
        Eigen::Vector2d newton_at_v = hessian_at_v.ldlt().solve(-grad_at_v);
        search_dir << newton_at_v(0), newton_at_v(1);
    }   
    // std::cout << "search_dir" << search_dir << std::endl;
    // do linesearch
    // std::cout << "local E0 = " << local_energy_0 << std::endl;
    // TODO: set to ls_param
    auto pos_copy = vertex_attrs[vid].pos;
    int max_itr = 200;
    double step = 1.0;
    double new_energy;
    auto new_x = uv_local;
    bool ls_good = false;
    for (int i = 0; i < max_itr; i++)
    {
        new_x.row(v_map[vid]) = uv_local.row(v_map[vid]) + step * search_dir;
        vertex_attrs[vid].pos << new_x(v_map[vid], 0), new_x(v_map[vid], 1);
        new_energy = compute_energy(new_x);
        // std::cout << "new E " << new_energy << std::endl;

        bool has_flip = false;
        for (auto loc : locs)
        {
            if (is_inverted(loc))
            {
                has_flip = true;
                break;
            }
        }
        if (new_energy < local_energy_0 && !has_flip)
        {
    
            ls_good = true;
            break;
        }
        step = step * 0.8;
    }
    if (ls_good)
    {
        wmtk::logger().trace("ls good, step = {}, energy {} -> {}", step, local_energy_0, new_energy);
    }
    else
    {
        wmtk::logger().trace("ls failed");
        vertex_attrs[vid].pos = pos_copy;
    }


    return ls_good;
}

void extremeopt::ExtremeOpt::smooth_all_vertices()
{
    igl::Timer timer;
    double time;
    timer.start();
    auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_vertices()) {
        collect_all_ops.emplace_back("vertex_smooth", loc);
    }
    
    // prepare to compute energy
    Eigen::SparseMatrix<double> G_global;
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    export_mesh(V, F, uv);
    get_grad_op(V, F, G_global);
    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);
    auto compute_energy = [&G_global, &dblarea](Eigen::MatrixXd &aaa)
    {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_global, aaa, Ji);
        return wmtk::compute_energy_from_jacobian(Ji, dblarea);
    };
    
    time = timer.getElapsedTime();
    wmtk::logger().info("vertex smoothing prepare time: {}s", time);
    wmtk::logger().debug("Num verts {}", collect_all_ops.size());
    // TODO: add N_iterations, E_target to ls-param
    int N_iters = 500;
    double E_target = 5.0;
    double E = compute_energy(uv);
    wmtk::logger().info("Start Energy E = {}", E);

    double E_old = E;
    for (int i = 1; i <= N_iters; i++)
    {
        if (NUM_THREADS > 0) {
            timer.start();
            auto executor = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kPartition>();
            executor.lock_vertices = [](auto& m, const auto& e, int task_id) -> bool {
                return m.try_set_vertex_mutex_one_ring(e, task_id);
            };
            executor.num_threads = NUM_THREADS;
            executor(*this, collect_all_ops);
            time = timer.getElapsedTime();
            wmtk::logger().info("vertex smoothing operation time parallel: {}s", time);
        } else {
            timer.start();
            auto executor = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
            executor(*this, collect_all_ops);
            time = timer.getElapsedTime();
            wmtk::logger().info("vertex smoothing operation time serial: {}s", time);
        }
        export_mesh(V, F, uv);
        E = compute_energy(uv);
        wmtk::logger().info("After Iter {}, E = {}", i, E);
        if (E < E_target)
        {
            wmtk::logger().info("Reach target energy({}), optimization succeed!", E_target);
            break;
        }
        if (E == E_old)
        {
            wmtk::logger().info("Energy get stuck, optimization failed.");
            break;
        }
        E_old = E;
    }
}

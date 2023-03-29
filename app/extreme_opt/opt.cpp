
#include "ExtremeOpt.h"
#include "wmtk/ExecutionScheduler.hpp"

#include <Eigen/src/Core/util/Constants.h>
#include <igl/Timer.h>

#include <Eigen/Sparse>
#include <array>
#include <wmtk/utils/Logger.hpp>
#include <wmtk/utils/TriQualityUtils.hpp>

#include <igl/upsample.h>
#include <igl/writeOBJ.h>
#include <limits>
#include <optional>
#include <wmtk/utils/TupleUtils.hpp>
#include "SYMDIR.h"

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
        if (m_params.save_meshes) igl::writeOBJ("new_tests/" + m_params.model_name + "_step_" + std::to_string(i) + "_splitted.obj", V, F, V, F, uv, F);

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
        if (m_params.save_meshes) igl::writeOBJ("new_tests/" + m_params.model_name + "_step_" + std::to_string(i) + "_swapped.obj", V, F, V, F, uv, F);

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
        if (m_params.save_meshes) igl::writeOBJ("new_tests/" + m_params.model_name + "_step_" + std::to_string(i) + "_collapsed.obj", V, F, V, F, uv, F);

        if (this->m_params.local_smooth) {
            timer.start();
            smooth_all_vertices();
            time = timer.getElapsedTime();
            wmtk::logger().info("LOCAL smoothing operation time serial: {}s", time);
            export_mesh(V, F, uv);
            get_grad_op(V, F, G_global);
            igl::doublearea(V, F, dblarea);
            E = compute_energy(uv);
            E_max = compute_energy_max(uv);
            wmtk::logger().info("After LOCAL smoothing {}, E = {}", i, E);
            wmtk::logger().info("E_max = {}", E_max);
            opt_log["opt_log"].push_back(
                {{"F_size", F.rows()}, {"V_size", V.rows()}, {"E_max", E_max}, {"E_avg", E}});
        }
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
        if (m_params.save_meshes) igl::writeOBJ("new_tests/" + m_params.model_name + "_step_" + std::to_string(i) + "_smoothed.obj", V, F, V, F, uv, F);

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

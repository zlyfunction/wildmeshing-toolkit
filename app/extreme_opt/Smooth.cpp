
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

#include <limits>
#include <optional>
#include <wmtk/utils/TupleUtils.hpp>
#include "SYMDIR.h"

#include <igl/writeOBJ.h>

using namespace wmtk;
auto renew = [](auto& m, auto op, auto& tris) {
    auto edges = m.new_edges_after(tris);
    auto optup = std::vector<std::pair<std::string, wmtk::TriMesh::Tuple>>();
    for (auto& e : edges) optup.emplace_back(op, e);
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

bool extremeopt::ExtremeOpt::swap_edge_before(const Tuple& t)
{
    
    if (!TriMesh::swap_edge_before(t)) return false;
    // if (vertex_attrs[t.vid(*this)].fixed &&
    // vertex_attrs[t.switch_vertex(*this).vid(*this)].fixed) return false;
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

bool extremeopt::ExtremeOpt::swap_edge_after(const Tuple& t)
{
    // std::cout << "connectivity test:" << std::endl;
    auto tri1 = oriented_tri_vids(t);
    auto t_opp = t.switch_face(*this);
    auto tri2 = oriented_tri_vids(t_opp.value());
    
    // std::cout << "trinagle 1 : ";
    // for (auto i : tri1) std::cout << i << " ";
    // std::cout << std::endl << "triangle 2 : ";
    // for (auto i : tri2) std::cout << i << " ";
    // std::cout << std::endl;

    std::vector<int> v_ids;
    Eigen::MatrixXi F_local(2, 3);
    Eigen::MatrixXd V_local(4, 3), uv_local(4, 2);
    for (int i = 0; i < 3; i++)
    {
        v_ids.push_back(tri1[i]);
        F_local(0, i) = i;
    }
    for (int i = 0; i < 3; i++)
    {
        int j = 0; 
        while (j < 3 && v_ids[j] != (int)tri2[i])
        {
            j++;
        }
        if ( j == 3) v_ids.push_back((int)tri2[i]);
        F_local(1, i) = j;
    }
    assert(v_ids.size() == 4);

    for (int i = 0; i < 4; i++)
    {
        V_local.row(i) = vertex_attrs[v_ids[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[v_ids[i]].pos;
    }

    Eigen::VectorXd dblarea, dblarea_3d;
    igl::doublearea(V_local, F_local, dblarea_3d);
    igl::doublearea(uv_local, F_local, dblarea);

    if (dblarea_3d.minCoeff() <= 0.0)
    {
        // std::cout << "zero 3d area: " << dblarea_3d.minCoeff() << std::endl;
        return false;
    }
    if (dblarea.minCoeff() <= 0.0)
    {
        // std::cout << "zero/negative 2d area: " << dblarea.minCoeff() << std::endl;
        return false;
    }



    return true;
}

bool extremeopt::ExtremeOpt::smooth_before(const Tuple& t)
{
    if (!t.is_valid(*this))
    {
        std::cout << "tuple not valid" << std::endl;
        return false;
    }
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
        area_local(i) = face_attrs[t_id].area_3d;
    }

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
    bool ls_good = false;
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

    auto collect_all_ops_swap = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_edges()) {
        collect_all_ops_swap.emplace_back("edge_swap", loc);
    }

    // prepare to compute energy
    Eigen::SparseMatrix<double> G_global;
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    export_mesh(V, F, uv);
    get_grad_op(V, F, G_global);
    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);
    auto compute_energy = [&G_global, &dblarea](Eigen::MatrixXd& aaa) {
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_global, aaa, Ji);

        return wmtk::compute_energy_from_jacobian(Ji, dblarea);
    };


    // auto setup_and_execute = [&](auto swap_executor) {
    //     // swap_executor.renew_neighbor_tuples = renew;
    //     swap_executor.num_threads = NUM_THREADS;
    //     swap_executor.priority = [](auto& m, auto op, const Tuple& e) {
    //         return m.compute_vertex_valence(e);
    //     };
    //     // swap_executor.lock_vertices = edge_locker;
    //     swap_executor.should_renew = [](auto val) { return (val > 0); };
    //     swap_executor.is_weight_up_to_date = [](auto& m, auto& ele) {
    //         auto& [val, _, e] = ele;
    //         auto val_energy = (m.compute_vertex_valence(e));
    //         return (val_energy > 1e-5) && ((val_energy - val) * (val_energy - val) < 1e-8);
    //     };
    //     swap_executor(*this, collect_all_ops);
    // };


    time = timer.getElapsedTime();
    wmtk::logger().info("vertex smoothing prepare time: {}s", time);
    wmtk::logger().debug("Num verts {}", collect_all_ops.size());
    double E = compute_energy(uv);
    wmtk::logger().info("Start Energy E = {}", E);

    double E_old = E;
    for (int i = 1; i <= m_params.max_iters; i++) {
        timer.start();
        auto executor = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
        executor(*this, collect_all_ops);
        time = timer.getElapsedTime();
        wmtk::logger().info("vertex smoothing operation time serial: {}s", time);

        export_mesh(V, F, uv);
        get_grad_op(V, F, G_global);
        igl::doublearea(V, F, dblarea);
        E = compute_energy(uv);
        wmtk::logger().info("After Iter {}, E = {}", i, E);
        igl::writeOBJ("mesh_after_smooth.obj", V, F, uv, F, uv, F);


        wmtk::logger().info("try swappping operations");

        auto setup_and_execute = [&](auto& executor_swap) {
            // executor_swap.renew_neighbor_tuples = renew;
            // executor_swap.priority = [&](auto& m, auto op, auto& t) { return m.get_length2(t); };
            executor_swap.num_threads = NUM_THREADS;
            executor_swap(*this, collect_all_ops_swap);
        };


        auto executor_swap = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
        setup_and_execute(executor_swap);

        export_mesh(V, F, uv);
        get_grad_op(V, F, G_global);

        igl::doublearea(V, F, dblarea);
        Eigen::VectorXd dblarea2d;
        igl::doublearea(uv, F, dblarea2d);
        // std::cout << "min area 3d\n" << dblarea.minCoeff() << std::endl;
        // std::cout << "area 2d" << dblarea2d << std::endl;
        for (int i = 0; i < F.rows(); i++)
        {
            face_attrs[i].area_3d = dblarea(i);
        }
        E = compute_energy(uv);
        wmtk::logger().info("After swapping, E = {}", E);
        igl::writeOBJ("mesh_after_swapping.obj", V, F, uv, F, uv, F);

        if (E < m_params.E_target) {
            wmtk::logger().info(
                "Reach target energy({}), optimization succeed!",
                m_params.E_target);
            break;
        }
        if (E == E_old) {
            wmtk::logger().info("Energy get stuck, optimization failed.");
            break;
        }
        E_old = E;
    }
}
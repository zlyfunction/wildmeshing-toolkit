
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

void extremeopt::ExtremeOpt::cache_edge_postions(const Tuple& t)
{
    position_cache.local().V1 = vertex_attrs[t.vid(*this)].pos_3d;
    position_cache.local().V2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos_3d;
    position_cache.local().uv1 = vertex_attrs[t.vid(*this)].pos;
    position_cache.local().uv2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos;

    double E1, E2;
    Eigen::MatrixXd V_local, uv_local, Ji;
    Eigen::MatrixXi F_local;
    Eigen::SparseMatrix<double> G_local;
    get_mesh_onering(t, V_local, uv_local, F_local);
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    E1 = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();

    get_mesh_onering(t.switch_vertex(*this), V_local, uv_local, F_local);
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    E1 = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();

    position_cache.local().E_max_before_collpase = std::max(E1, E2);
    std::cout << "local E_max before collapse: " << position_cache.local().E_max_before_collpase << std::endl;
}   

bool extremeopt::ExtremeOpt::collapse_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) 
    {
        std::cout << "not valid" << std::endl;
        return false;
    }
    if (!wmtk::TriMesh::collapse_edge_before(t)) 
    {
        std::cout << "link cond fail" << std::endl;
        return false;
    }
    // if (vertex_attrs[t.vid(*this)].fixed && vertex_attrs[t.switch_vertex(*this).vid(*this)].fixed) return false;
    cache_edge_postions(t);
    return true;
}

bool extremeopt::ExtremeOpt::collapse_edge_after(const Tuple& t)
{
    const Eigen::Vector3d V = (position_cache.local().V1 + position_cache.local().V2) / 2.0;
    const Eigen::Vector2d uv = (position_cache.local().uv1 + position_cache.local().uv2) / 2.0;
    auto vid = t.vid(*this);
    vertex_attrs[vid].pos_3d = V;
    vertex_attrs[vid].pos = uv;
    
    // get local F,V,uv
    auto vid_onering = get_one_ring_vids_for_vertex(vid);
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
    if (area_local_3d.minCoeff() <= 0 || area_local.minCoeff() <= 0)
    {
        std::cout << "collapse causing flips" << std::endl;
        return false;
    }

    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);

    double E_max_after_collapse;
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    E_max_after_collapse = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();
    
    if (E_max_after_collapse >= position_cache.local().E_max_before_collpase)
    {
        // E_max does not go down
        return false;
    }
    
    std::cout << "collapse succeed" << std::endl;
    return true;
}
bool extremeopt::ExtremeOpt::swap_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) return false;
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


    // TODO: compute and compare local energy
    // get the F_local_before swapping
    // get dblareas, then compute the energy
    Eigen::MatrixXi F_local_old(2, 3);
    int v_to_change = 0;
    while (F_local(1, 0) != v_to_change && F_local(1, 1) != v_to_change && F_local(1, 2) != v_to_change)
    {
        v_to_change++;
    }
    // std::cout << "change: " << v_to_change << "->3" << std::endl;
    // std::cout << "change: " << (v_to_change + 1) % 3 << "<->" << (v_to_change + 2) % 3 << std::endl;

    for (int i = 0; i < 3; i++)
    {
        if (F_local(0, i) == v_to_change)
        {
            F_local_old(0, i) = 3;
        }
        else
        {
            F_local_old(0, i) = F_local(0, i);
        }
    }

    for (int i = 0; i < 3; i++)
    {
        if (F_local(1, i) == (v_to_change + 1) % 3)
        {
            F_local_old(1, i) = (v_to_change + 2) % 3;
        }
        else if (F_local(1, i) == (v_to_change + 2) % 3)
        {
            F_local_old(1, i) = (v_to_change + 1) % 3;
        }
        else
        {
            F_local_old(1, i) = F_local(1, i);
        }
    }

    // std::cout << "F_local after swap:\n" << F_local << std::endl;
    // std::cout << "F_local before swap:\n" << F_local_old << std::endl;
    
    Eigen::VectorXd dblarea_3d_old;
    igl::doublearea(V_local, F_local_old, dblarea_3d_old);
    
    Eigen::SparseMatrix<double> G_local, G_local_old;
    get_grad_op(V_local, F_local, G_local);
    get_grad_op(V_local, F_local_old, G_local_old);

    double E, E_old;
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    
    E = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff(); // compute E_max
    // E = wmtk::compute_energy_from_jacobian(Ji, dblarea_3d) * dblarea_3d.sum(); // compute E_sum
    wmtk::jacobian_from_uv(G_local_old, uv_local, Ji);
    
    E_old = wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff(); // compute E_max
    // E_old = wmtk::compute_energy_from_jacobian(Ji, dblarea_3d_old) * dblarea_3d_old.sum(); // compute E_sum

    // std::cout << "energy before swap: " << E_old << std::endl;
    // std::cout << "energy after swap: " << E << std::endl;

    if (E_old < E)
    {
        // std::cout << "energy increase after swapping" << std::endl;
        return false;
    }
    else
    {
        // std::cout << "energy decreased, do swap" << std::endl;
    }


    // if do swap, change the areas
    face_attrs[t.fid(*this)].area_3d = dblarea_3d[0];
    face_attrs[t_opp.value().fid(*this)].area_3d = dblarea_3d[1];
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
        executor_swap.num_threads = NUM_THREADS;
        executor_swap(*this, collect_all_ops_swap);
    };
    auto executor_swap = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    setup_and_execute(executor_swap);
}

void extremeopt::ExtremeOpt::collapse_all_edges()
{
    auto collect_all_ops_collapse = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_edges())
    {
        collect_all_ops_collapse.emplace_back("edge_collapse", loc);
    }
    auto setup_and_execute = [&](auto& executor_collapse) {
        // executor_collapse.renew_neighbor_tuples = renew;
        executor_collapse.num_threads = NUM_THREADS;
        executor_collapse(*this, collect_all_ops_collapse);
    };
    auto executor_collapse = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    setup_and_execute(executor_collapse);
}
void extremeopt::ExtremeOpt::do_optimization()
{
    igl::Timer timer;
    double time;
    
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
    auto compute_energy_max = [&G_global, &dblarea](Eigen::MatrixXd& aaa){
        Eigen::MatrixXd Ji;
        wmtk::jacobian_from_uv(G_global, aaa, Ji);
        return wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();
    };

    double E = compute_energy(uv);
    wmtk::logger().info("Start Energy E = {}", E);

    double E_old = E;
    for (int i = 1; i <= m_params.max_iters; i++) 
    {
        // do smoothing
timer.start();
        // smooth_all_vertices();
time = timer.getElapsedTime();
        wmtk::logger().info("vertex smoothing operation time serial: {}s", time);
        export_mesh(V, F, uv);
        get_grad_op(V, F, G_global);
        igl::doublearea(V, F, dblarea);
        E = compute_energy(uv);
        wmtk::logger().info("After smoothing {}, E = {}", i, E);

        // do swaping
timer.start();
        if (this->m_params.do_swap) {
            swap_all_edges();
        }
time = timer.getElapsedTime();
        wmtk::logger().info("edges swapping operation time serial: {}s", time);
        export_mesh(V, F, uv);

        get_grad_op(V, F, G_global);
        igl::doublearea(V, F, dblarea);
        Eigen::VectorXd dblarea2d;
        igl::doublearea(uv, F, dblarea2d);
        // std::cout << "min area 3d: " << dblarea.minCoeff() << std::endl;
        // std::cout << "min area 2d: " << dblarea2d.minCoeff() << std::endl;
        E = compute_energy(uv);
        wmtk::logger().info("After swapping, E = {}", E);
        wmtk::logger().info("E_max = {}", compute_energy_max(uv));

        // TODO: add other operations
        if (this->m_params.do_collapse)
        {
            collapse_all_edges();
        }
       
        export_mesh(V, F, uv);
        get_grad_op(V, F, G_global);
        igl::doublearea(V, F, dblarea);
        igl::doublearea(uv, F, dblarea2d);
        // std::cout << "min area 3d: " << dblarea.minCoeff() << std::endl;
        // std::cout << "min area 2d: " << dblarea2d.minCoeff() << std::endl;
        E = compute_energy(uv);
        wmtk::logger().info("Mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());
        wmtk::logger().info("After collapsing, E = {}", E);
        wmtk::logger().info("E_max = {}", compute_energy_max(uv));

        // terminate criteria
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
        std::cout << std::endl;

    }

}
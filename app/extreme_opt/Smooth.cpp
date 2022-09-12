
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
    
    std::cout << "vertex " << vid << std::endl;

    auto vid_onering = get_one_ring_vids_for_vertex(vid);
    auto locs = get_one_ring_tris_for_vertex(t);
    assert(locs.size() > 0);
    
    // get local (V, F)
    Eigen::MatrixXd V_local(vid_onering.size(), 3);
    for (size_t i = 0; i < vid_onering.size(); i++)
    {
        V_local.row(i) = vertex_attrs[vid_onering[i]].pos_3d;
    }
    std::vector<int> v_map(vertex_attrs.size(), -1);
    for (size_t i = 0; i < vid_onering.size(); i++)
    {
        v_map[vid_onering[i]] = i;
    }
    Eigen::MatrixXi F_local(locs.size(), 3);
    for (size_t i = 0; i < locs.size(); i++)
    {
        int t_id = locs[i].fid(*this);
        auto local_tuples = oriented_tri_vertices(locs[i]);
        for (size_t j = 0; j < 3; j++)
        {
            F_local(i, j) = v_map[local_tuples[j].vid(*this)];
        }
    }

    Eigen::SparseMatrix<double> G_local;
    get_grad_op(V_local, F_local, G_local);
    std::cout << "G_local: \n" << G_local << std::endl;

    // std::cout << "V_local:\n" << V_local << "F_local:\n" << F_local << std::endl;
    return true;
    
    
    // Computes the maximal error around the one ring
    // that is needed to ensure the operation will decrease the error measure
    auto max_quality = 0.;
    for (auto& tri : locs) {
        max_quality = std::max(max_quality, get_quality(tri));
    }



    assert(max_quality > 0); // If max quality is zero it is likely that the triangles are flipped

    // Collects the coordinate of all vertices in the 1-ring
    std::vector<std::array<double, 6>> assembles(locs.size());
    auto loc_id = 0;

    // For each triangle, make a reordered copy of the vertices so that
    // the vertex to optimize is always the first
    for (auto& loc : locs) {
        auto& T = assembles[loc_id];
        auto t_id = loc.fid(*this);

        assert(!is_inverted(loc));
        auto local_tuples = oriented_tri_vertices(loc);
        std::array<size_t, 3> local_verts;
        for (auto i = 0; i < 3; i++) {
            local_verts[i] = local_tuples[i].vid(*this);
        }

        local_verts = wmtk::orient_preserve_tri_reorder(local_verts, vid);

        for (auto i = 0; i < 3; i++) {
            for (auto j = 0; j < 2; j++) {
                T[i * 2 + j] = vertex_attrs[local_verts[i]].pos[j];
            }
        }
        loc_id++;
    }

    // Make a backup of the current configuration
    auto old_pos = vertex_attrs[vid].pos;
    auto old_asssembles = assembles;

    // Minimize distortion using newton's method
    vertex_attrs[vid].pos = wmtk::newton_method_from_stack_2d(
        assembles,
        wmtk::AMIPS2D_energy,
        wmtk::AMIPS2D_jacobian,
        wmtk::AMIPS2D_hessian);

    // Logging
    wmtk::logger().info(
        "old pos {} -> new pos {}",
        old_pos.transpose(),
        vertex_attrs[vid].pos.transpose());

    return true;
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
    time = timer.getElapsedTime();
    wmtk::logger().info("vertex smoothing prepare time: {}s", time);
    wmtk::logger().debug("Num verts {}", collect_all_ops.size());
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
}

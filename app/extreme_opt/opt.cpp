
#include "CollapsePairOperation.h"
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
#include "SYMDIR_NEW.h"

#include <paraviewo/HDF5VTUWriter.hpp>
#include <paraviewo/ParaviewWriter.hpp>
#include <paraviewo/VTUWriter.hpp>
using namespace wmtk;


void extremeopt::ExtremeOpt::do_optimization(json& opt_log)
{
    igl::Timer timer;
    double time;

    int split_succ_cnt_cache = m_params.split_succ_cnt;


    // get edge length thresholds for collapsing operation
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    export_mesh(V, F, uv);
    elen_threshold = 0;
    elen_threshold_3d = 0;
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            int v1 = F(i, j);
            int v2 = F(i, (j + 1) % 3);
            double elen = (uv.row(v1) - uv.row(v2)).norm();
            double elen_3d = (V.row(v1) - V.row(v2)).norm();
            if (elen > elen_threshold) elen_threshold = elen;
            if (elen_3d > elen_threshold_3d) elen_threshold_3d = elen_3d;
        }
    }
    elen_threshold *= m_params.elen_alpha;
    elen_threshold_3d *= m_params.elen_alpha;
    
    double E = get_quality();
    wmtk::logger().info("Start Energy E = {}", E);

    double E_max = get_quality_max();
    wmtk::logger().info("Start E_max = {}", E_max);
    
    opt_log["opt_log"].push_back(
                {{"F_size", get_faces().size()}, {"V_size", get_vertices().size()}, {"E_max", E_max}, {"E", E}});
    double E_old = E;
    int V_size, F_size;

    if (m_params.save_meshes)
    {
        export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus/", m_params.model_name + std::to_string(0) + ".hdf");
    }

    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_max;
        
        if (m_params.save_meshes)
        {
            export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus_detail/", m_params.model_name + std::to_string(i)+"_0" + ".hdf");
        }
        if (this->m_params.do_split) {

            if (i <= 2)
            {
                m_params.split_succ_cnt = 100000;
            }
            else
            {
                m_params.split_succ_cnt = split_succ_cnt_cache;
            }
            // TODO: set priority inside split_all_edges
            auto Es = get_quality_all();
            timer.start();
            split_all_edges(Es);
            time = timer.getElapsedTime();
            wmtk::logger().info("edges splitting operation time serial: {}s", time);

            E = get_quality();
            E_max = get_quality_max();

            V_size = get_vertices().size();
            F_size = get_faces().size();

            wmtk::logger()
                .info("Mesh F size {}, V size {}", F_size, V_size);
            wmtk::logger().info("After splitting, E = {}", E);
            wmtk::logger().info("E_max = {}",E_max);
            spdlog::info("E is {} {} {}", std::isfinite(E), !std::isnan(E), !std::isinf(E));
        }
        if (m_params.save_meshes)
        {
            export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus_detail/", m_params.model_name + std::to_string(i)+"_1" + ".hdf");
        }

        if (this->m_params.do_swap) {
            timer.start();
            swap_all_edges();
            time = timer.getElapsedTime();
            wmtk::logger().info("edges swapping operation time serial: {}s", time);

            E = get_quality();
            E_max = get_quality_max();

            V_size = get_vertices().size();
            F_size = get_faces().size();

            wmtk::logger()
                .info("Mesh F size {}, V size {}", F_size, V_size);
            wmtk::logger().info("After swapping, E = {}", E);
            wmtk::logger().info("E_max = {}",E_max);
            spdlog::info("E is {} {} {}", std::isfinite(E), !std::isnan(E), !std::isinf(E));
        }
        if (m_params.save_meshes)
        {
            export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus_detail/", m_params.model_name + std::to_string(i)+"_2" + ".hdf");
        }
        
        if (this->m_params.do_collapse) {
            timer.start();
            collapse_all_edges();
            time = timer.getElapsedTime();
            wmtk::logger().info("edges collapsing operation time serial: {}s", time);

            E = get_quality();
            E_max = get_quality_max();

            V_size = get_vertices().size();
            F_size = get_faces().size();

            wmtk::logger()
                .info("Mesh F size {}, V size {}", F_size, V_size);
            wmtk::logger().info("After collapsing, E = {}", E);
            wmtk::logger().info("E_max = {}",E_max);
            spdlog::info("E is {} {} {}", std::isfinite(E), !std::isnan(E), !std::isinf(E));
        }
        if (m_params.save_meshes)
        {
            export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus_detail/", m_params.model_name + std::to_string(i)+"_3" + ".hdf");
        }

        if (this->m_params.local_smooth) {
            timer.start();
            smooth_all_vertices();
            time = timer.getElapsedTime();
            wmtk::logger().info("LOCAL smoothing operation time serial: {}s", time);

            E = get_quality();
            E_max = get_quality_max();

            wmtk::logger().info("After LOCAL smoothing {}, E = {}", i, E);
            wmtk::logger().info("E_max = {}", E_max);
        }
        if (this->m_params.global_smooth) {
            timer.start();
            smooth_global(1);
            time = timer.getElapsedTime();
            wmtk::logger().info("GLOBAL smoothing operation time serial: {}s", time);
            
            E = get_quality();
            E_max = get_quality_max();

            wmtk::logger().info("After GLOBAL smoothing {}, E = {}", i, E);
            wmtk::logger().info("E_max = {}", E_max);
        }
        opt_log["opt_log"].push_back(
                {{"F_size", F_size}, {"V_size", V_size}, {"E_max", E_max}, {"E", E}});
        if (m_params.save_meshes)
        {
            export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus_detail/", m_params.model_name + std::to_string(i)+"_4" + ".hdf");
        }
        // TODO: terminate criteria
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
        if (m_params.save_meshes)
        {
            export_mesh_vtu("/home/leyi/wildmeshing-toolkit/build/new_tests/vtus/", m_params.model_name + std::to_string(i) + ".hdf");
        }
    }
}

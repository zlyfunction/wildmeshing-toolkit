#include "ExtremeOpt.h"
#include <spdlog/common.h>
#include <CLI/CLI.hpp>

#include <igl/read_triangle_mesh.h>
#include <igl/boundary_loop.h>

int main(int argc, char** argv)
{
    ZoneScopedN("extreme_opt_main");

    CLI::App app{argv[0]};
    std::string input_file = "./";
    std::string output_file = "./";

    extremeopt::Parameters param;
    app.add_option("-i,--input", input_file, "Input mesh.");
    app.add_option("-o,--output", output_file, "Output mesh.");
    app.add_option("--max-its", param.max_iters, "max iters");
    app.add_option("--E-target", param.E_target, "target energy");
    app.add_option("--ls-its", param.ls_iters, "linesearch max iterations, min-stepsize=0.8^{ls-its}");
    app.add_option("--do-newton", param.do_newton, "do newton or do gradient descent");
    app.add_option("--do-swap", param.do_swap, "do swaps or not");
    app.add_option("--do-collapse", param.do_collapse, "do collapse or not");
    app.add_option("--split-thresh", param.split_thresh, "split length threshold");
    // app.add_option("-j,--jobs", NUM_THREADS, "thread.");

    CLI11_PARSE(app, argc, argv);

    // Loading the input mesh
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    igl::readOBJ(input_file, V, uv, uv, F, F, F);
    wmtk::logger().info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());
    
    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);
    int Nv_bds = 0;
    for (auto bd : bds) Nv_bds += bd.size();
    wmtk::logger().info("Boundary size: {}", Nv_bds);
    // Load the mesh in the trimesh class
    extremeopt::ExtremeOpt extremeopt;
    extremeopt.create_mesh(V,F,uv);
    extremeopt.m_params = param;

    assert(extremeopt.check_mesh_connectivity_validity());

    // TODO: do smoothing
    // extremeopt.smooth_all_vertices();
    extremeopt.do_optimization();
    extremeopt.write_obj("after_collpase.obj");
    // Do the mesh optimization
    // extremeopt.optimize();
    // extremeopt.consolidate_mesh();

    // Save the optimized mesh
    // extremeopt.write_mesh(output_file);

    // Output
    // auto [max_energy, avg_energy] = mesh.get_max_avg_energy();
    // std::ofstream fout(output_file + ".log");
    // fout << "#t: " << mesh.tet_size() << std::endl;
    // fout << "#v: " << mesh.vertex_size() << std::endl;
    // fout << "max_energy: " << max_energy << std::endl;
    // fout << "avg_energy: " << avg_energy << std::endl;
    // fout << "eps: " << params.eps << std::endl;
    // fout << "threads: " << NUM_THREADS << std::endl;
    // fout << "time: " << time << std::endl;
    // fout.close();

    // igl::write_triangle_mesh(output_path + "_surface.obj", matV, matF);
    // wmtk::logger().info("Output face size {}", outface.size());
    // wmtk::logger().info("======= finish =========");

    return 0;
}
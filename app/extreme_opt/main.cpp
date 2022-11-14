#include "ExtremeOpt.h"
#include <spdlog/common.h>
#include <CLI/CLI.hpp>
#include <fstream>

#include <igl/read_triangle_mesh.h>
#include <igl/boundary_loop.h>
#include <igl/upsample.h>
#include <igl/writeOBJ.h>
#include "json.hpp"
using json = nlohmann::json;

int main(int argc, char** argv)
{
    ZoneScopedN("extreme_opt_main");

    CLI::App app{argv[0]};
    std::string input_dir = "./objs";
    std::string output_dir = "./test_out";
    std::string input_json = "../config/config.json";
    std::string model = "knot1";
    extremeopt::Parameters param;
    app.add_option("-i,--input", input_dir, "Input mesh dir.");
    app.add_option("-m,--model", model, "Input model name.");
    app.add_option("-j,--json", input_json, "Input arguments.");
    app.add_option("-o,--output", output_dir, "Output dir.");
    // app.add_option("--max-its", param.max_iters, "max iters");
    // app.add_option("--E-target", param.E_target, "target energy");
    // app.add_option("--ls-its", param.ls_iters, "linesearch max iterations, min-stepsize=0.8^{ls-its}");
    // app.add_option("--do-newton", param.do_newton, "do newton or do gradient descent");
    // app.add_option("--do-swap", param.do_swap, "do swaps or not");
    // app.add_option("--do-collapse", param.do_collapse, "do collapse or not");
    // app.add_option("--split-thresh", param.split_thresh, "split length threshold");
    // app.add_option("-j,--jobs", NUM_THREADS, "thread.");

    CLI11_PARSE(app, argc, argv);

    std::string input_file = input_dir + "/" + model + "_init.obj";
    // Loading the input mesh
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    igl::readOBJ(input_file, V, uv, uv, F, F, F);
    wmtk::logger().info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());
    
    std::ifstream js_in(input_json); json config = json::parse(js_in);
    param.max_iters     = config["max_iters"];
    param.E_target      = config["E_target"];
    param.ls_iters      = config["ls_iters"];
    param.do_newton     = config["do_newton"];
    param.do_collapse   = config["do_collapse"];
    param.do_swap       = config["do_swap"];

    json opt_log;
    opt_log["model_name"] = model;
    opt_log["args"] = config;
    std::ofstream js_out(output_dir + "/" + model + ".json");
    
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

  
    extremeopt.do_optimization(opt_log);
    extremeopt.export_mesh(V, F, uv);
    for (int i = 0; i < 5; i++)
    {
        std::cout << "do upsample" << std::endl;
        Eigen::MatrixXi new_F;
        Eigen::MatrixXd new_V, new_uv;
        igl::upsample(V, F, new_V, new_F);
        igl::upsample(uv, F, new_uv, new_F);
        std::cout << "F size " << F.rows() << " --> " << new_F.rows() << std::endl;
        std::cout << "V size " << V.rows() << " --> " << new_V.rows() << std::endl;

        extremeopt::ExtremeOpt extremeopt1;
        extremeopt1.create_mesh(new_V,new_F,new_uv);
        extremeopt1.m_params = param;
        extremeopt1.do_optimization(opt_log);
        extremeopt1.export_mesh(V, F, uv);
    }
    igl::writeOBJ(output_dir + "/" + model + "_out.obj", V, F, V, F, uv, F);
    js_out << std::setw(4) << opt_log << std::endl;
    
    // extremeopt.write_obj("after_collpase.obj");
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
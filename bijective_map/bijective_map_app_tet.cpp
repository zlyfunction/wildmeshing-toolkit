#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
// wmtk
#include <wmtk/TetMesh.hpp>
#include <wmtk/TriMesh.hpp>
#include <wmtk/io/MeshReader.hpp>
// Application modules
#include "csv_io.hpp"
#include "tet_curve_tracking.hpp"
#include "tet_point_tracking_app.hpp"
#include "tet_surface_sampling.hpp"
#include "tet_surface_tracking.hpp"
#include "tet_surface_tracking_with_connectivity.hpp"
#include "vtu_utils.hpp"

using path = std::filesystem::path;
using json = nlohmann::json;

int main(int argc, char** argv)
{
    CLI::App app{"bijective_map_app_tet"};
    std::filesystem::path config_file;
    app.add_option("-c, --config", config_file, "JSON config file (overrides other options)");
    // Default values
    std::filesystem::path initial_mesh_file;
    std::filesystem::path operation_logs_dir;
    std::filesystem::path output_mesh_file;
    std::string application_name = "back";
    std::filesystem::path surface_file = "query_surface.json";
    size_t start_operation = 0;
    size_t save_interval = 1;
    std::filesystem::path save_dir;
    bool do_rounding = false;
    bool do_simplify = false;
    bool only_do_arrangement_once = false;
    // CLI options (used if no config file)
    app.add_option("-a, --app", application_name, "Application name");
    app.add_option("-i, --input", initial_mesh_file, "Initial mesh file");
    app.add_option("-l, --logs", operation_logs_dir, "Operation logs directory");
    app.add_option("-o, --output-mesh", output_mesh_file, "Output mesh file");
    app.add_option("-s, --surface", surface_file, "Surface file");
    app.add_option("--start-op", start_operation, "Start from operation N (default: 0)");
    app.add_option(
        "--save-interval",
        save_interval,
        "Save surface every N operations (default: 1)");
    app.add_option("--save-dir", save_dir, "Directory to save intermediate surfaces");
    app.add_option("--do-rounding", do_rounding, "Round barycentric coordinates");
    app.add_option("--do-simplify", do_simplify, "Simplify refined triangles");
    app.add_option(
        "--only-do-arrangement-once",
        only_do_arrangement_once,
        "Perform final autorefine arrangement on the before mesh (default: false)");
    CLI11_PARSE(app, argc, argv);
    // If config file is provided, read parameters from JSON
    if (!config_file.empty()) {
        std::cout << "Reading config from: " << config_file << std::endl;
        std::ifstream ifs(config_file);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open config file: " << config_file << std::endl;
            return 1;
        }
        json config = json::parse(ifs);
        if (config.contains("application_name"))
            application_name = config["application_name"].get<std::string>();
        if (config.contains("initial_mesh_file"))
            initial_mesh_file = config["initial_mesh_file"].get<std::string>();
        if (config.contains("operation_logs_dir"))
            operation_logs_dir = config["operation_logs_dir"].get<std::string>();
        if (config.contains("output_mesh_file"))
            output_mesh_file = config["output_mesh_file"].get<std::string>();
        if (config.contains("surface_file"))
            surface_file = config["surface_file"].get<std::string>();
        if (config.contains("start_operation"))
            start_operation = config["start_operation"].get<size_t>();
        if (config.contains("save_interval")) save_interval = config["save_interval"].get<size_t>();
        if (config.contains("save_dir")) save_dir = config["save_dir"].get<std::string>();
        if (config.contains("do_rounding")) do_rounding = config["do_rounding"].get<bool>();
        if (config.contains("do_simplify")) do_simplify = config["do_simplify"].get<bool>();
        if (config.contains("only_do_arrangement_once"))
            only_do_arrangement_once = config["only_do_arrangement_once"].get<bool>();
    }
    // Validate required parameters
    if (initial_mesh_file.empty() || operation_logs_dir.empty() || output_mesh_file.empty()) {
        std::cerr
            << "Error: initial_mesh_file, operation_logs_dir, and output_mesh_file are required"
            << std::endl;
        return 1;
    }
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "  application_name: " << application_name << std::endl;
    std::cout << "  initial_mesh_file: " << initial_mesh_file << std::endl;
    std::cout << "  operation_logs_dir: " << operation_logs_dir << std::endl;
    std::cout << "  output_mesh_file: " << output_mesh_file << std::endl;
    std::cout << "  surface_file: " << surface_file << std::endl;
    std::cout << "  start_operation: " << start_operation << std::endl;
    std::cout << "  save_interval: " << save_interval << std::endl;
    std::cout << "  save_dir: " << save_dir << std::endl;
    std::cout << "  do_rounding: " << do_rounding << std::endl;
    std::cout << "  do_simplify: " << do_simplify << std::endl;
    std::cout << "  only_do_arrangement_once: " << only_do_arrangement_once << std::endl;
    auto init_mesh_ptr = wmtk::read_mesh(initial_mesh_file);
    // Get T_before and V_before from init_mesh_ptr using get_TV()
    std::cout << "\n=== Reading T_before and V_before from init_mesh_ptr ===" << std::endl;
    auto [T_before, V_before] = static_cast<wmtk::TetMesh&>(*init_mesh_ptr).get_TV();
    std::cout << "T_before: " << T_before.rows() << " x " << T_before.cols() << std::endl;
    std::cout << "V_before: " << V_before.rows() << " x " << V_before.cols() << std::endl;
    // Write init_mesh to VTU file for visualization
    std::cout << "\n=== Writing init_mesh to VTU file ===" << std::endl;
    std::filesystem::path init_mesh_vtu = initial_mesh_file.filename();
    init_mesh_vtu.replace_extension(".vtu");
    vtu_utils::write_tet_mesh_to_vtu(V_before, T_before, init_mesh_vtu.string());
    std::cout << "Successfully wrote init_mesh to: " << init_mesh_vtu << std::endl;
    // Read mesh data from VTU file
    std::cout << "\n=== Reading T_after and V_after from out vtu file ===" << std::endl;
    Eigen::MatrixXd V_after;
    Eigen::MatrixXi T_after;
    bool vtu_success_after =
        vtu_utils::read_tet_mesh_from_vtu(output_mesh_file.string(), V_after, T_after);
    if (vtu_success_after) {
        std::cout << "Successfully read output mesh file" << std::endl;
        std::cout << "  Vertices: " << V_after.rows() << " x " << V_after.cols() << std::endl;
        std::cout << "  Tetrahedra: " << T_after.rows() << " x " << T_after.cols() << std::endl;
    } else {
        std::cerr << "Failed to read output mesh file" << std::endl;
        return 1;
    }
    std::cout << "\n=== Running application ===" << std::endl;
    if (application_name == "back") {
        std::string output_points_file = output_mesh_file.stem().string() + "_points.vtu";
        std::string initial_points_file = initial_mesh_file.stem().string() + "_points.vtu";
        tet_point_tracking::run_back_tracking(
            T_after,
            V_after,
            V_before,
            operation_logs_dir,
            output_points_file,
            initial_points_file);
    } else if (application_name == "back_r") {
        std::string output_points_file = output_mesh_file.stem().string() + "_points_rational.vtu";
        std::string initial_points_file =
            initial_mesh_file.stem().string() + "_points_rational.vtu";
        tet_point_tracking::run_back_tracking_rational(
            T_after,
            V_after,
            V_before,
            operation_logs_dir,
            output_points_file,
            initial_points_file);
    } else if (application_name == "back_curve") {
        tet_curve_tracking::run_back_tracking_curve(T_after, V_after, V_before, operation_logs_dir);
    } else if (application_name == "back_surface") {
        tet_surface_tracking::run_back_tracking_surface(
            T_after,
            V_after,
            V_before,
            operation_logs_dir,
            surface_file,
            false);
    } else if (application_name == "back_surface_connectivity") {
        tet_surface_tracking_with_connectivity::run_backward_tracking_surface(
            T_after,
            V_after,
            T_before,
            V_before,
            operation_logs_dir,
            surface_file,
            false,
            start_operation,
            save_interval,
            save_dir,
            do_rounding,
            do_simplify,
            only_do_arrangement_once);
    }
    return 0;
}

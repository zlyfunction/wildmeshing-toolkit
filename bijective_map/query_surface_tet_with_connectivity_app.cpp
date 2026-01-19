#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <wmtk/TetMesh.hpp>
#include <wmtk/io/MeshReader.hpp>
#include <igl/read_triangle_mesh.h>

#include "tet_surface_sampling.hpp"
#include "tet_surface_tracking_with_connectivity.hpp"

using json = nlohmann::json;
using path = std::filesystem::path;

int main(int argc, char** argv)
{
    CLI::App app{"query_surface_tet_with_connectivity_app"};

    path config_file;
    path tet_mesh_file;
    path surface_mesh_file;
    path output_json;
    path output_vtu;
    double tolerance = 1e-8;
    bool verbose = false;

    app.add_option("-c, --config", config_file, "JSON config file (overrides other options)");
    app.add_option("-t, --tet", tet_mesh_file, "Input tet mesh file (.msh)");
    app.add_option("-s, --surface", surface_mesh_file, "Input surface mesh file (.stl/.obj/.off)");
    app.add_option("--out-json", output_json, "Output JSON file");
    app.add_option("--out-vtu", output_vtu, "Output VTU file");
    app.add_option("--tol", tolerance, "Tolerance for point-in-tet search");
    app.add_flag("--verbose", verbose, "Verbose autorefine output");

    CLI11_PARSE(app, argc, argv);

    if (!config_file.empty()) {
        std::ifstream ifs(config_file);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open config file: " << config_file << std::endl;
            return 1;
        }
        json config = json::parse(ifs);
        if (config.contains("tet_mesh_file")) {
            tet_mesh_file = config["tet_mesh_file"].get<std::string>();
        }
        if (config.contains("surface_mesh_file")) {
            surface_mesh_file = config["surface_mesh_file"].get<std::string>();
        }
        if (config.contains("output_json")) {
            output_json = config["output_json"].get<std::string>();
        }
        if (config.contains("output_vtu")) {
            output_vtu = config["output_vtu"].get<std::string>();
        }
        if (config.contains("tolerance")) {
            tolerance = config["tolerance"].get<double>();
        }
        if (config.contains("verbose")) {
            verbose = config["verbose"].get<bool>();
        }
    }

    if (tet_mesh_file.empty() || surface_mesh_file.empty()) {
        std::cerr << "Error: --tet and --surface are required" << std::endl;
        return 1;
    }

    if (output_json.empty() || output_vtu.empty()) {
        path base_dir = surface_mesh_file.parent_path();
        std::string stem = surface_mesh_file.stem().string();
        if (output_json.empty()) {
            output_json = base_dir / (stem + "_query_surface_tet_with_connectivity.json");
        }
        if (output_vtu.empty()) {
            output_vtu = base_dir / (stem + "_query_surface_tet_with_connectivity.vtu");
        }
    }

    std::cout << "Reading tet mesh: " << tet_mesh_file << std::endl;
    auto mesh_ptr = wmtk::read_mesh(tet_mesh_file);
    if (!mesh_ptr) {
        std::cerr << "Failed to read tet mesh: " << tet_mesh_file << std::endl;
        return 1;
    }
    auto [T, V] = static_cast<wmtk::TetMesh&>(*mesh_ptr).get_TV();
    std::cout << "  Tets: " << T.rows() << ", Vertices: " << V.rows() << std::endl;

    std::cout << "Reading surface mesh: " << surface_mesh_file << std::endl;
    Eigen::MatrixXd V_surface;
    Eigen::MatrixXi F_surface;
    if (!igl::read_triangle_mesh(surface_mesh_file.string(), V_surface, F_surface)) {
        std::cerr << "Failed to read surface mesh: " << surface_mesh_file << std::endl;
        return 1;
    }
    std::cout << "  Surface vertices: " << V_surface.rows()
              << ", Surface triangles: " << F_surface.rows() << std::endl;

    auto query_surface =
        tet_surface_sampling::query_surface_tet_with_connectivity_from_triangle_mesh(
            T,
            V,
            V_surface,
            F_surface,
            tolerance,
            verbose);

    std::cout << "Writing JSON: " << output_json << std::endl;
    tet_surface_tracking_with_connectivity::write_surface_connectivity_to_file(
        query_surface,
        output_json.string());

    std::cout << "Writing VTU: " << output_vtu << std::endl;
    tet_surface_tracking_with_connectivity::write_surface_to_vtu(
        query_surface,
        V,
        output_vtu.string());

    return 0;
}

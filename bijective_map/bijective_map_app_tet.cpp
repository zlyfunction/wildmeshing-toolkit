#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
// wmtk
#include <wmtk/TetMesh.hpp>
#include <wmtk/TriMesh.hpp>
#include <wmtk/io/MeshReader.hpp>

// Application modules
#include "csv_io.hpp"
#include "tet_point_tracking.hpp"
#include "tet_curve_tracking.hpp"
#include "tet_surface_tracking.hpp"
#include "tet_surface_sampling.hpp"

using path = std::filesystem::path;

int main(int argc, char** argv)
{
    CLI::App app{"bijective_map_app_tet"};
    std::filesystem::path initial_mesh_file;
    std::filesystem::path operation_logs_dir;
    std::string application_name = "back";
    app.add_option("-a, --app", application_name, "Application name");
    app.add_option("-i, --input", initial_mesh_file, "Initial mesh file")->required(true);
    app.add_option("-l, --logs", operation_logs_dir, "Operation logs directory")->required(true);

    std::filesystem::path surface_file = "query_surface.json";
    app.add_option("-s, --surface", surface_file, "Surface file");
    CLI11_PARSE(app, argc, argv);

    std::cout << "Application name: " << application_name << std::endl;
    auto init_mesh_ptr = wmtk::read_mesh(initial_mesh_file);

    // write initial mesh to vtu
    // std::cout << "Writing initial mesh to vtu" << std::endl;
    // wmtk::io::ParaviewWriter
    //     writer("initial_mesh", "vertices", *init_mesh_ptr, true, true, true, true);
    // init_mesh_ptr->serialize(writer);
    // TODO:
    // 2. figure out how to read the outputmesh in vtu format out is after remesh
    auto T_after = csv_io::readTetrahedrons("../build/T_matrix_out.csv");
    auto V_after = csv_io::readVertices("../build/V_matrix_out.csv");
    // // TODO: for now, first we convert it with TV matrix
    auto T_before = csv_io::readTetrahedrons("../build/T_matrix_in.csv");
    auto V_before = csv_io::readVertices("../build/V_matrix_in.csv");

    std::cout << "T_after.rows(): " << T_after.rows() << std::endl;
    std::cout << "V_after.rows(): " << V_after.rows() << std::endl;
    std::cout << "T_before.rows(): " << T_before.rows() << std::endl;
    std::cout << "V_before.rows(): " << V_before.rows() << std::endl;

    if (application_name == "back") {
        tet_point_tracking::run_back_tracking(T_after, V_after, V_before, operation_logs_dir);
    } else if (application_name == "back_curve") {
        tet_curve_tracking::run_back_tracking_curve(T_after, V_after, V_before, operation_logs_dir);
    } else if (application_name == "back_surface") {
        tet_surface_tracking::run_back_tracking_surface(T_after, V_after, V_before, operation_logs_dir, surface_file, false);
    }

    return 0;
}

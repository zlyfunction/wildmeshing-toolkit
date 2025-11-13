#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
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

int main(int argc, char** argv)
{
    CLI::App app{"bijective_map_app_tet"};
    std::filesystem::path initial_mesh_file;
    std::filesystem::path operation_logs_dir;
    std::filesystem::path output_mesh_file;

    std::string application_name = "back";
    app.add_option("-a, --app", application_name, "Application name");
    app.add_option("-i, --input", initial_mesh_file, "Initial mesh file")->required(true);
    app.add_option("-l, --logs", operation_logs_dir, "Operation logs directory")->required(true);
    app.add_option("-o, --output-mesh", output_mesh_file, "Output mesh file")->required(true);

    std::filesystem::path surface_file = "query_surface.json";
    app.add_option("-s, --surface", surface_file, "Surface file");
    CLI11_PARSE(app, argc, argv);

    std::cout << "Application name: " << application_name << std::endl;
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
    std::cout << "✓ Successfully wrote init_mesh to: " << init_mesh_vtu << std::endl;

    // Read mesh data from VTU file
    std::cout << "\n=== Reading T_after and V_after from out vtu file ===" << std::endl;
    Eigen::MatrixXd V_after;
    Eigen::MatrixXi T_after;

    bool vtu_success_after =
        vtu_utils::read_tet_mesh_from_vtu(output_mesh_file.string(), V_after, T_after);
    if (vtu_success_after) {
        std::cout << "✓ Successfully read output mesh file" << std::endl;
        std::cout << "  Vertices: " << V_after.rows() << " x " << V_after.cols() << std::endl;
        std::cout << "  Tetrahedra: " << T_after.rows() << " x " << T_after.cols() << std::endl;
    } else {
        std::cerr << "✗ Failed to read output mesh file" << std::endl;
    }

    // ===== OLD CSV-based approach (commented out) =====
    // // Read mesh data from CSV files
    // std::cout << "\n=== Reading from CSV files ===" << std::endl;
    // auto T_after_csv = csv_io::readTetrahedrons("../build/T_matrix_out.csv");
    // auto V_after_csv = csv_io::readVertices("../build/V_matrix_out.csv");
    // auto T_before_csv = csv_io::readTetrahedrons("../build/T_matrix_in.csv");
    // auto V_before_csv = csv_io::readVertices("../build/V_matrix_in.csv");
    //
    // std::cout << "CSV T_after: " << T_after_csv.rows() << " x " << T_after_csv.cols() <<
    // std::endl; std::cout << "CSV V_after: " << V_after_csv.rows() << " x " << V_after_csv.cols()
    // << std::endl; std::cout << "CSV T_before: " << T_before_csv.rows() << " x " <<
    // T_before_csv.cols() << std::endl; std::cout << "CSV V_before: " << V_before_csv.rows() << " x
    // " << V_before_csv.cols() << std::endl;

    std::cout << "\n=== Running application ===" << std::endl;

    if (application_name == "back") {
        // Generate filenames based on mesh files (without path, only filename)
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
        // Generate filenames for rational version
        std::string output_points_file = output_mesh_file.stem().string() + "_points_rational.vtu";
        std::string initial_points_file = initial_mesh_file.stem().string() + "_points_rational.vtu";
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
            V_before,
            operation_logs_dir,
            surface_file,
            false);
    }

    return 0;
}

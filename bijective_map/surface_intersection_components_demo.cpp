#include "surface_intersection_components.hpp"
#include "tet_surface_tracking_with_connectivity.hpp"
#include "vtu_utils.hpp"

#include <Eigen/Core>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
struct Options
{
    std::filesystem::path tet_vtu;
    std::filesystem::path surface_a;
    std::filesystem::path surface_b;
    bool verbose = false;
};

void print_usage(const char* exe)
{
    std::cout << "Usage: " << exe
              << " --tet_vtu <mesh.vtu> --surface_a <surf_a.json> --surface_b <surf_b.json>"
              << " [--verbose]\n";
}

Options parse_args(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tet_vtu" && i + 1 < argc) {
            opts.tet_vtu = argv[++i];
        } else if (arg == "--surface_a" && i + 1 < argc) {
            opts.surface_a = argv[++i];
        } else if (arg == "--surface_b" && i + 1 < argc) {
            opts.surface_b = argv[++i];
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (opts.tet_vtu.empty() || opts.surface_a.empty() || opts.surface_b.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("Missing required arguments.");
    }
    return opts;
}

tet_surface_tracking_with_connectivity::MatrixXr to_rational_vertices(
    const Eigen::MatrixXd& V_double)
{
    tet_surface_tracking_with_connectivity::MatrixXr V_rational(V_double.rows(), V_double.cols());
    for (Eigen::Index i = 0; i < V_double.rows(); ++i) {
        for (Eigen::Index j = 0; j < V_double.cols(); ++j) {
            V_rational(i, j) = wmtk::Rational(V_double(i, j), false);
        }
    }
    return V_rational;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opts = parse_args(argc, argv);

        Eigen::MatrixXd V_double;
        Eigen::MatrixXi T;
        if (!vtu_utils::read_tet_mesh_from_vtu(opts.tet_vtu.string(), V_double, T)) {
            throw std::runtime_error("Failed to read tet mesh from " + opts.tet_vtu.string());
        }
        if (V_double.cols() != 3) {
            throw std::runtime_error("Tet mesh vertices must have 3 columns.");
        }

        const auto surface_a =
            tet_surface_tracking_with_connectivity::read_surface_connectivity_from_file(
                opts.surface_a.string());
        const auto surface_b =
            tet_surface_tracking_with_connectivity::read_surface_connectivity_from_file(
                opts.surface_b.string());

        const auto V_rational = to_rational_vertices(V_double);
        const auto result =
            tet_surface_tracking_with_connectivity::compute_surface_intersection_components(
                surface_a,
                surface_b,
                V_rational,
                opts.verbose);

        std::cout << "Intersection components: " << result.component_count << "\n";
        std::cout << "Intersection segments: " << result.segment_count << "\n";
        std::cout << "Intersection points: " << result.point_count << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#include "tet_surface_tracking_with_connectivity.hpp"
#include <CGAL/number_utils.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "batch_operation_log_reader.hpp"
#include "cgal_autorefine_utils_rational.hpp"
#include "tet_point_tracking.hpp"
#include "tet_surface_sampling.hpp"
#include "tet_surface_tracking_internal.hpp"
#include "tet_track_operations.hpp"
#include "tet_track_operations_internal.hpp"
#include "vtu_utils.hpp"

namespace tet_surface_tracking_with_connectivity {


void write_surface_to_vtu_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V,
    const std::string& filename)
{
    std::cout << "Writing surface with connectivity to VTU file: " << filename << std::endl;

    // First convert to world positions
    auto [V_rational, F] = surface_to_world_positions_rational(query_surface, V);

    // Convert rational coordinates to double for VTU output
    Eigen::MatrixXd V_double(V_rational.rows(), V_rational.cols());
    for (int i = 0; i < V_rational.rows(); i++) {
        for (int j = 0; j < V_rational.cols(); j++) {
            V_double(i, j) = V_rational(i, j).to_double();
        }
    }

    // Prepare tet_id as cell scalar (one value per triangle)
    Eigen::VectorXi tet_ids_vector(query_surface.tet_ids.size());
    for (size_t i = 0; i < query_surface.tet_ids.size(); i++) {
        tet_ids_vector(i) = query_surface.tet_ids[i];
    }

    // Write to VTU using existing utility function
    vtu_utils::write_triangle_mesh_to_vtu(V_double, F, filename, &tet_ids_vector, "tet_id");

    std::cout << "Successfully wrote " << V_double.rows() << " vertices and " << F.rows()
              << " triangles to " << filename << std::endl;
    std::cout << "  with tet_id property attached to each triangle" << std::endl;
}

void write_surface_to_vtu(
    const query_surface_tet_with_connectivity& query_surface,
    const Eigen::MatrixXd& V,
    const std::string& filename)
{
    // Convert double matrix to rational matrix
    MatrixXr V_rational(V.rows(), V.cols());
    for (int i = 0; i < V.rows(); i++) {
        for (int j = 0; j < V.cols(); j++) {
            V_rational(i, j) = wmtk::Rational(V(i, j));
        }
    }

    // Call rational version
    write_surface_to_vtu_rational(query_surface, V_rational, filename);
}


void run_backward_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold)
{
    std::cout << "Backward tracking surface with connectivity" << std::endl;

    // Step 1: Read or sample the query surface with connectivity
    std::string query_surface_filename = surface_file.string();
    query_surface_tet_with_connectivity query_surface;
    if (!std::filesystem::exists(query_surface_filename)) {
        std::cout << "query_surface not found, sampling and writing to file..." << std::endl;
        // User must provide surface file or use external sampling functions
        // query_surface =
        // tet_surface_sampling::sample_query_surface_tet_with_connectivity(T_after, V_after);
        int N = 5; // number of slicing planes (can be parameterized)
        int axis = 2; // slice axis (can be parameterized)
        double min_coord = V_after.col(axis).minCoeff();
        double max_coord = V_after.col(axis).maxCoeff();
        double plane_coord =
            min_coord + (max_coord - min_coord) * (N + 1) / (N + 2); // single point near last slice
        query_surface = tet_surface_sampling::slice_tet_mesh_with_axis_plane(
            T_after,
            V_after,
            axis,
            plane_coord);
        write_surface_connectivity_to_file(query_surface, query_surface_filename);
    } else {
        std::cout << "query_surface found, reading from file..." << std::endl;
        query_surface = read_surface_connectivity_from_file(query_surface_filename);
    }

    std::string model_name = "model";
    auto dir_str = operation_logs_dir.filename().string();
    const std::string prefix = "operation_log_";
    if (dir_str.size() > prefix.size() && dir_str.substr(0, prefix.size()) == prefix) {
        model_name = dir_str.substr(prefix.size());
    }
    write_surface_to_vtu(
        query_surface,
        V_after,
        model_name + "_query_surface_tet_with_connectivity_after.vtu");

    {
        // DEBUG: sanitity check for the input query_surface
        for (int i = 0; i < query_surface.query_triangles.size(); i++) {
            const Eigen::Vector3i& tri = query_surface.query_triangles[i];
            int tri_tet_id = query_surface.tet_ids[i];

            const Eigen::Vector4i& relevant_vids = T_after.row(tri_tet_id);

            // check points
            for (int j = 0; j < 3; j++) {
                const auto& pt = query_surface.points[tri[j]];
                if (pt.t_id != tri_tet_id) {
                    std::cout << "i: " << i << ", tri_tet_id: " << tri_tet_id
                              << ", pt: [t_id=" << pt.t_id << ", bc=(" << pt.bc[0].to_double()
                              << ", " << pt.bc[1].to_double() << ", " << pt.bc[2].to_double()
                              << ", " << pt.bc[3].to_double() << ")]" << std::endl;

                    const auto tv_ids = pt.tv_ids;
                    for (int bc_idx = 0; bc_idx < 4; bc_idx++) {
                        if (pt.bc(bc_idx) != 0) {
                            int v_idx = tv_ids(bc_idx);
                            if (std::find(relevant_vids.data(), relevant_vids.data() + 4, v_idx) ==
                                relevant_vids.data() + 4) {
                                std::cout << "ERROR: " << "v_idx: " << v_idx
                                          << " is not in relevant_vids" << std::endl;
                                std::cout << "bc of this point: " << pt.bc(bc_idx).to_double()
                                          << std::endl;
                                throw std::runtime_error("Error: point not in relevant_vids");
                            }
                        }
                    }
                }
            }
        }
    }

    // step2 do the backward tracking
    std::cout << "Doing backward tracking..." << std::endl;
    track_all_operations(operation_logs_dir, query_surface, false);
    std::cout << "Backward tracking completed" << std::endl;

    // step3 write the surface to file
    write_surface_connectivity_to_file(
        query_surface,
        model_name + "_query_surface_tet_with_connectivity_before.json");
    write_surface_to_vtu(
        query_surface,
        V_before,
        model_name + "_query_surface_tet_with_connectivity_before.vtu");

    // TODO: results sanity check
}

void write_surface_connectivity_to_file(
    const query_surface_tet_with_connectivity& surface,
    const std::string& filename)
{
    std::cout << "Writing surface with connectivity to file: " << filename << std::endl;

    json j;
    j["num_points"] = surface.points.size();
    j["num_triangles"] = surface.query_triangles.size();

    // Write all points
    for (size_t i = 0; i < surface.points.size(); ++i) {
        const auto& pt = surface.points[i];
        json pt_json;
        pt_json["t_id"] = pt.t_id;

        // Store barycentric coordinates as strings to preserve rational precision
        pt_json["bc"] = {
            pt.bc[0].serialize(),
            pt.bc[1].serialize(),
            pt.bc[2].serialize(),
            pt.bc[3].serialize()};

        // Store tetrahedron vertex ids
        pt_json["tv_ids"] = {pt.tv_ids[0], pt.tv_ids[1], pt.tv_ids[2], pt.tv_ids[3]};

        j["points"].push_back(pt_json);
    }

    // Write all triangles
    for (size_t i = 0; i < surface.query_triangles.size(); ++i) {
        const auto& tri = surface.query_triangles[i];
        json tri_json;
        tri_json["indices"] = {tri[0], tri[1], tri[2]};

        if (i < surface.tet_ids.size()) {
            tri_json["tet_id"] = surface.tet_ids[i];
        }

        j["triangles"].push_back(tri_json);
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
        std::cout << "Successfully wrote " << surface.points.size() << " points and "
                  << surface.query_triangles.size() << " triangles to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

query_surface_tet_with_connectivity read_surface_connectivity_from_file(const std::string& filename)
{
    std::cout << "Reading surface with connectivity from file: " << filename << std::endl;
    query_surface_tet_with_connectivity surface;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return surface;
    }

    json j;
    file >> j;
    file.close();

    size_t num_points = j["num_points"];
    size_t num_triangles = j["num_triangles"];

    surface.points.resize(num_points);
    surface.query_triangles.resize(num_triangles);
    surface.tet_ids.resize(num_triangles);

    // Read all points
    for (size_t i = 0; i < num_points; ++i) {
        const auto& pt_json = j["points"][i];
        auto& pt = surface.points[i];

        pt.t_id = pt_json["t_id"];

        // Read barycentric coordinates from serialized strings using deserialize
        const auto& bc_array = pt_json["bc"];
        pt.bc = Eigen::Matrix<wmtk::Rational, 4, 1>(
            wmtk::Rational::deserialize(bc_array[0]),
            wmtk::Rational::deserialize(bc_array[1]),
            wmtk::Rational::deserialize(bc_array[2]),
            wmtk::Rational::deserialize(bc_array[3]));

        // Read tetrahedron vertex ids
        const auto& tv_array = pt_json["tv_ids"];
        pt.tv_ids = Eigen::Vector4i(tv_array[0], tv_array[1], tv_array[2], tv_array[3]);
    }

    // Read all triangles
    for (size_t i = 0; i < num_triangles; ++i) {
        const auto& tri_json = j["triangles"][i];
        auto& tri = surface.query_triangles[i];

        const auto& indices = tri_json["indices"];
        tri = Eigen::Vector3i(indices[0], indices[1], indices[2]);

        if (tri_json.contains("tet_id")) {
            surface.tet_ids[i] = tri_json["tet_id"];
        }
    }

    std::cout << "Successfully read " << surface.points.size() << " points and "
              << surface.query_triangles.size() << " triangles from " << filename << std::endl;

    return surface;
}

} // namespace tet_surface_tracking_with_connectivity

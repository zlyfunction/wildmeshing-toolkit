#include "tet_surface_tracking_with_connectivity.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "batch_operation_log_reader.hpp"
#include "tet_point_tracking.hpp"
#include "tet_surface_sampling.hpp"
#include "tet_track_operations.hpp"
#include "tet_track_operations_internal.hpp"
#include "vtu_utils.hpp"

namespace tet_surface_tracking_with_connectivity {

std::pair<MatrixXr, Eigen::MatrixXi> surface_to_world_positions_rational(
    const query_surface_tet_with_connectivity& query_surface,
    const MatrixXr& V)
{
    // TODO: Implement conversion from query surface to world positions
    MatrixXr V_out;
    Eigen::MatrixXi F_out;

    return {V_out, F_out};
}

void print_surface_area_statistics(const MatrixXr& surface_V, const Eigen::MatrixXi& surface_F)
{
    // TODO: Implement triangle area statistics calculation and printing
    std::cout << "\n=== Surface Area Statistics ===" << std::endl;
    std::cout << "Number of triangles: " << surface_F.rows() << std::endl;
    std::cout << "================================\n" << std::endl;
}

void check_surface_manifold_property(const MatrixXr& surface_V, const Eigen::MatrixXi& surface_F)
{
    // TODO: Implement manifold property checking
    std::cout << "Checking manifold property..." << std::endl;
}

void run_backward_tracking_surface(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::filesystem::path& surface_file,
    bool check_manifold)
{
    // TODO: Implement backward tracking surface application
    std::cout << "Backward tracking surface with connectivity" << std::endl;

    // Step 1: Read or sample the query surface with connectivity
    std::string query_surface_filename = surface_file.string();
    query_surface_tet_with_connectivity query_surface;
    if (!std::filesystem::exists(query_surface_filename)) {
        std::cout << "query_surface not found, sampling and writing to file..." << std::endl;
        // User must provide surface file or use external sampling functions
        query_surface =
            tet_surface_sampling::sample_query_surface_tet_with_connectivity(T_after, V_after);
        write_surface_connectivity_to_file(query_surface, query_surface_filename);
    } else {
        std::cout << "query_surface found, reading from file..." << std::endl;
        query_surface = read_surface_connectivity_from_file(query_surface_filename);
    }
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
            pt.bc[3].serialize()
        };

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

        // Read barycentric coordinates from serialized strings
        const auto& bc_array = pt_json["bc"];
        std::string bc0_str = bc_array[0];
        std::string bc1_str = bc_array[1];
        std::string bc2_str = bc_array[2];
        std::string bc3_str = bc_array[3];
        pt.bc = Eigen::Matrix<wmtk::Rational, 4, 1>(
            wmtk::Rational(bc0_str),
            wmtk::Rational(bc1_str),
            wmtk::Rational(bc2_str),
            wmtk::Rational(bc3_str)
        );

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

void handle_consolidate_operation(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet_with_connectivity& surface,
    bool forward)
{
    // TODO: Implement consolidate operation handling
    std::cout << "Handling Consolidate operation for surface with connectivity" << std::endl;
}

void handle_local_mapping_operation(
    const MatrixXr& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const MatrixXr& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_surface_tet_with_connectivity& surface)
{
    // TODO: Implement local mapping operation handling
    std::cout << "Handling Local Mapping operation for surface with connectivity" << std::endl;
}

void track_one_operation(
    const nlohmann::json& operation_log,
    query_surface_tet_with_connectivity& surface,
    bool do_forward,
    int operation_id)
{
    // TODO: Implement single operation tracking
    std::string operation_name = operation_log["operation_name"];
    std::cout << "Tracking operation: " << operation_name << " (ID: " << operation_id << ")"
              << std::endl;
}

void track_all_operations(
    const std::filesystem::path& dirPath,
    query_surface_tet_with_connectivity& surface,
    bool do_forward)
{
    // TODO: Implement tracking through all operations
    std::cout << "Tracking all operations from directory: " << dirPath << std::endl;

    BatchOperationLogReader reader(dirPath);
    size_t total_ops = reader.get_total_operations();

    if (total_ops == 0) {
        std::cerr << "No operation logs found in " << dirPath << std::endl;
        return;
    }

    std::cout << "Found " << total_ops << " operations in "
              << (reader.is_batch_format() ? "batch" : "legacy") << " format" << std::endl;

    // TODO: Iterate through operations and track each one
}

std::pair<std::vector<int>, std::vector<Vector4r>> get_point_representations(
    const int local_t_id,
    const Vector4r& local_bc,
    const Eigen::MatrixXi& T_local)
{
    // TODO: Implement getting all possible representations of a point
    std::vector<int> all_possible_t_ids;
    std::vector<Vector4r> all_possible_bcs;

    // Add the original representation
    all_possible_t_ids.push_back(local_t_id);
    all_possible_bcs.push_back(local_bc);

    return {all_possible_t_ids, all_possible_bcs};
}

} // namespace tet_surface_tracking_with_connectivity

#include "tet_surface_tracking_with_connectivity.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "batch_operation_log_reader.hpp"
#include "cgal_autorefine_utils_rational.hpp"
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
    std::cout << "Converting query surface with connectivity to world positions..." << std::endl;

    // Allocate output matrices
    // V_out: one row per unique point (not 3 per triangle)
    MatrixXr V_out(query_surface.points.size(), 3);
    Eigen::MatrixXi F_out(query_surface.query_triangles.size(), 3);

    // Convert each point to world coordinates
    for (size_t i = 0; i < query_surface.points.size(); i++) {
        const auto& pt = query_surface.points[i];

        // Get vertices of the tetrahedron
        Eigen::Vector4i tet_verts = pt.tv_ids;

        // Calculate real position using barycentric coordinates
        Vector3r world_pos = Vector3r::Zero();
        for (int j = 0; j < 4; j++) {
            world_pos += pt.bc(j) * V.row(tet_verts(j)).transpose();
        }
        V_out.row(i) = world_pos.transpose();
    }

    // Copy triangle connectivity
    for (size_t i = 0; i < query_surface.query_triangles.size(); i++) {
        F_out.row(i) = query_surface.query_triangles[i];
    }

    std::cout << "Converted " << V_out.rows() << " vertices and " << F_out.rows() << " triangles"
              << std::endl;

    return {V_out, F_out};
}

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

// TODO: Implement this!!!!
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
        query_surface =
            tet_surface_sampling::sample_query_surface_tet_with_connectivity(T_after, V_after);
        write_surface_connectivity_to_file(query_surface, query_surface_filename);
    } else {
        std::cout << "query_surface found, reading from file..." << std::endl;
        query_surface = read_surface_connectivity_from_file(query_surface_filename);
    }
    write_surface_to_vtu(query_surface, V_after, "query_surface_tet_with_connectivity_after.vtu");

    // step2 do the backward tracking
    std::cout << "Doing backward tracking..." << std::endl;
    track_all_operations(operation_logs_dir, query_surface, false);
    std::cout << "Backward tracking completed" << std::endl;

    // step3 write the surface to file
    write_surface_connectivity_to_file(
        query_surface,
        "query_surface_tet_with_connectivity_before.json");
    write_surface_to_vtu(query_surface, V_before, "query_surface_tet_with_connectivity_before.vtu");

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

void handle_consolidate_operation(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_surface_tet_with_connectivity& surface,
    bool forward)
{
    std::cout << "Handling Consolidate operation for surface with connectivity" << std::endl;

    // Step 1: Handle points using handle_consolidate_tet
    tet_point_tracking::handle_consolidate_tet<wmtk::Rational>(
        tet_ids_maps,
        vertex_ids_maps,
        surface.points,
        forward);

    // Step 2: Handle tet_ids (treat them like t_id in query_point_tet)
    if (!forward) {
        // Backward: direct mapping
        for (auto& tet_id : surface.tet_ids) {
            if (tet_id >= 0) {
                tet_id = tet_ids_maps[tet_id];
            }
        }
    } else {
        // Forward: search for the old value in the map
        for (auto& tet_id : surface.tet_ids) {
            if (tet_id >= 0) {
                auto it = std::find(tet_ids_maps.begin(), tet_ids_maps.end(), tet_id);
                if (it != tet_ids_maps.end()) {
                    tet_id = std::distance(tet_ids_maps.begin(), it);
                }
            }
        }
    }

    std::cout << "Consolidate operation completed for " << surface.points.size() << " points and "
              << surface.tet_ids.size() << " triangle tet_ids" << std::endl;
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

    // step1:map all points in the surface
    tet_point_tracking::handle_local_mapping_tet_exact(
        V_before,
        T_before,
        id_map_before,
        v_id_map_before,
        V_after,
        T_after,
        id_map_after,
        v_id_map_after,
        surface.points,
        true);

    // step2: get all faces in surface.triangle that is in id_map_after
    {
        std::vector<int> face_ids;
        for (int i = 0; i < surface.query_triangles.size(); i++) {
            if (std::find(id_map_after.begin(), id_map_after.end(), surface.tet_ids[i]) !=
                id_map_after.end()) {
                face_ids.push_back(i);
            }
        }

        // Step 2.1: Collect all unique point indices used by selected triangles
        std::set<int> unique_point_indices_set;
        for (int face_id : face_ids) {
            const Eigen::Vector3i& tri = surface.query_triangles[face_id];
            unique_point_indices_set.insert(tri[0]);
            unique_point_indices_set.insert(tri[1]);
            unique_point_indices_set.insert(tri[2]);
        }

        // Convert set to vector for indexing
        std::vector<int> unique_point_indices(
            unique_point_indices_set.begin(),
            unique_point_indices_set.end());

        // Step 2.2: Create mapping from global point index to local point index
        std::map<int, int> global_to_local_point_map;
        for (int local_idx = 0; local_idx < unique_point_indices.size(); local_idx++) {
            global_to_local_point_map[unique_point_indices[local_idx]] = local_idx;
        }

        // Step 2.3: Build local_triangles_F by remapping triangle indices to local indices
        Eigen::MatrixXi local_triangles_F(face_ids.size(), 3);
        for (int i = 0; i < face_ids.size(); i++) {
            int face_id = face_ids[i];
            const Eigen::Vector3i& global_tri = surface.query_triangles[face_id];

            // Map each vertex from global to local index
            local_triangles_F(i, 0) = global_to_local_point_map[global_tri[0]];
            local_triangles_F(i, 1) = global_to_local_point_map[global_tri[1]];
            local_triangles_F(i, 2) = global_to_local_point_map[global_tri[2]];
        }

        std::cout << "Built local triangles mesh: " << unique_point_indices.size() << " vertices, "
                  << local_triangles_F.rows() << " faces" << std::endl;

        // Step 2.4: Prepare sampled points for autorefine
        std::vector<cgal_autorefine_demo::SampledPointInputRational> sampled_points;
        sampled_points.reserve(unique_point_indices.size());

        for (int local_idx = 0; local_idx < unique_point_indices.size(); local_idx++) {
            int global_idx = unique_point_indices[local_idx];
            const auto& pt = surface.points[global_idx];

            cgal_autorefine_demo::SampledPointInputRational sampled_pt;

            // Find the position of pt.t_id in id_map_before
            auto it = std::find(id_map_before.begin(), id_map_before.end(), pt.t_id);
            if (it != id_map_before.end()) {
                sampled_pt.tet_index = std::distance(id_map_before.begin(), it);
            } else {
                // If not found in id_map_before, use -1 or handle error
                std::cerr << "Warning: tet_id " << pt.t_id << " not found in id_map_before" << std::endl;
                sampled_pt.tet_index = -1;
            }

            sampled_pt.barycentric = pt.bc;

            sampled_points.push_back(sampled_pt);
        }

        // Step 2.5: Call autorefine_sampled_triangles_rational on V_before and T_before
        std::cout << "Calling autorefine_sampled_triangles_rational on V_before and T_before..." << std::endl;
        cgal_autorefine_demo::AutorefineResultRational autorefine_result =
            cgal_autorefine_demo::autorefine_sampled_triangles_rational(
                V_before,
                T_before,
                sampled_points,
                local_triangles_F);

        std::cout << "Autorefine completed: "
                  << autorefine_result.refined_points.size() << " refined points, "
                  << autorefine_result.refined_triangles.size() << " refined triangles" << std::endl;
        std::cout << "Sampled fragment indices size: "
                  << autorefine_result.sampled_fragment_indices.size() << std::endl;
    }
}

void track_one_operation(
    const nlohmann::json& operation_log,
    query_surface_tet_with_connectivity& surface,
    bool do_forward,
    int operation_id)
{
    std::string operation_name = operation_log["operation_name"];
    std::cout << "Tracking operation: " << operation_name << " (ID: " << operation_id << ")"
              << std::endl;

    if (operation_name == "MeshConsolidate") {
        std::cout << "  This operation is Consolidate" << std::endl;
        std::vector<int64_t> tet_ids_maps;
        std::vector<int64_t> vertex_ids_maps;
        parse_consolidate_file_tet(operation_log, tet_ids_maps, vertex_ids_maps);

        handle_consolidate_operation(tet_ids_maps, vertex_ids_maps, surface, do_forward);
    } else {
        std::cout << "  This operation is " << operation_name << std::endl;

        // Parse operation data
        Eigen::MatrixXi T_after, T_before;
        Eigen::MatrixXd V_after_double, V_before_double;
        std::vector<int64_t> id_map_after, id_map_before;
        std::vector<int64_t> v_id_map_after, v_id_map_before;

        parse_non_collapse_file_tet(
            operation_log,
            V_before_double,
            T_before,
            id_map_before,
            v_id_map_before,
            V_after_double,
            T_after,
            id_map_after,
            v_id_map_after,
            operation_id);

        // Convert double matrices to rational matrices
        MatrixXr V_before(V_before_double.rows(), V_before_double.cols());
        MatrixXr V_after(V_after_double.rows(), V_after_double.cols());

        for (int i = 0; i < V_before_double.rows(); i++) {
            for (int j = 0; j < V_before_double.cols(); j++) {
                V_before(i, j) = wmtk::Rational(V_before_double(i, j));
            }
        }

        for (int i = 0; i < V_after_double.rows(); i++) {
            for (int j = 0; j < V_after_double.cols(); j++) {
                V_after(i, j) = wmtk::Rational(V_after_double(i, j));
            }
        }

        // Call handle_local_mapping_operation with appropriate direction
        if (do_forward) {
            handle_local_mapping_operation(
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                surface);
        } else {
            handle_local_mapping_operation(
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                surface);
        }
    }

    std::cout << "  Operation " << operation_id << " completed" << std::endl;
}

void track_all_operations(
    const std::filesystem::path& dirPath,
    query_surface_tet_with_connectivity& surface,
    bool do_forward)
{
    std::cout << "Tracking all operations from directory: " << dirPath << std::endl;

    BatchOperationLogReader reader(dirPath);
    size_t total_ops = reader.get_total_operations();

    if (total_ops == 0) {
        std::cerr << "No operation logs found in " << dirPath << std::endl;
        return;
    }

    std::cout << "Found " << total_ops << " operations in "
              << (reader.is_batch_format() ? "batch" : "legacy") << " format" << std::endl;

    // Iterate through operations in the appropriate order
    for (size_t i = 0; i < total_ops; ++i) {
        size_t operation_index = i;
        if (!do_forward) {
            // Backward tracking: process operations in reverse order
            operation_index = total_ops - 1 - i;
        }

        json operation_log = reader.get_operation(operation_index);
        if (operation_log.empty()) {
            std::cerr << "Failed to read operation " << operation_index << std::endl;
            continue;
        }

        std::cout << "\n=== Processing operation " << (i + 1) << "/" << total_ops
                  << " (index: " << operation_index << ") ===" << std::endl;

        track_one_operation(operation_log, surface, do_forward, static_cast<int>(operation_index));
    }

    std::cout << "\n=== All operations completed ===" << std::endl;
    std::cout << "Final surface state: " << surface.points.size() << " points, "
              << surface.query_triangles.size() << " triangles" << std::endl;
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

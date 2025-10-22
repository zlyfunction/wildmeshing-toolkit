#include "tet_point_tracking.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include "tet_track_operations.hpp"
#include "vtu_utils.hpp"
#include "tet_track_operations_internal.hpp"
#include "batch_operation_log_reader.hpp"
#include "FindPointTetMesh.hpp"
#include <wmtk/utils/Rational.hpp>

namespace tet_point_tracking {

Eigen::MatrixXd write_points_to_file(
    const std::vector<query_point_tet>& query_points,
    const Eigen::MatrixXd& V,
    const std::string& filename)
{
    // Compute point coordinates and store in Eigen::MatrixXd
    Eigen::MatrixXd point_coords(query_points.size(), 3);

    for (int i = 0; i < query_points.size(); i++) {
        auto& qp = query_points[i];
        Eigen::Vector3d p(0, 0, 0);
        for (int j = 0; j < 4; j++) {
            p += qp.bc(j) * V.row(qp.tv_ids[j]);
        }
        point_coords.row(i) = p;
    }

    // Write to VTU file using vtu_utils
    vtu_utils::write_point_mesh_to_vtu(point_coords, filename);

    return point_coords;
}

void run_back_tracking(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir,
    const std::string& points_after_remesh_filename,
    const std::string& points_after_tracking_filename)
{
    std::cout << "Back tracking" << std::endl;

    // Sample points in T_after, V_after
    std::vector<query_point_tet> query_points;

    // Sample some points on boundary tetrahedrons
    std::vector<query_point_tet> boundary_query_points;
    std::unordered_map<std::string, int> face_count;

    // Count the occurrence of each face
    for (int i = 0; i < T_after.rows(); i++) {
        auto tet = T_after.row(i);
        for (int j = 0; j < 4; j++) {
            Eigen::Vector3i face;
            face << tet[j], tet[(j + 1) % 4], tet[(j + 2) % 4];
            std::sort(face.data(), face.data() + 3);
            std::string face_key = std::to_string(face[0]) + "_" + std::to_string(face[1]) + "_" +
                                   std::to_string(face[2]);
            face_count[face_key]++;
        }
    }

    // Find boundary tetrahedrons and sample
    for (int i = 0; i < T_after.rows(); i++) {
        if (i % 3 != 0) continue;
        auto tet = T_after.row(i);
        int boundary_face_count = 0;
        for (int j = 0; j < 4; j++) {
            Eigen::Vector3i face;
            face << tet[j], tet[(j + 1) % 4], tet[(j + 2) % 4];
            std::sort(face.data(), face.data() + 3);
            std::string face_key = std::to_string(face[0]) + "_" + std::to_string(face[1]) + "_" +
                                   std::to_string(face[2]);
            if (face_count[face_key] == 1) {
                boundary_face_count++;
            }
        }
        if (boundary_face_count > 0) {
            query_point_tet qp;
            qp.t_id = i;
            qp.bc = Eigen::Vector4d::Random().cwiseAbs(); // Randomize barycentric coordinates
            qp.bc /= qp.bc.sum(); // Normalize to ensure they sum to 1
            qp.tv_ids = T_after.row(i);
            boundary_query_points.push_back(qp);
        }
    }

    // Add sampled points from boundary tetrahedrons to the total query points
    query_points.insert(
        query_points.end(),
        boundary_query_points.begin(),
        boundary_query_points.end());

    // compute position and save to file
    std::cout << "Writing points to file after remesh" << std::endl;
    auto points_before = write_points_to_file(query_points, V_after, points_after_remesh_filename);
    track_point_tet(operation_logs_dir, query_points, false, false);

    std::cout << "Writing points to file after back tracking" << std::endl;
    auto points_after =
        write_points_to_file(query_points, V_before, points_after_tracking_filename);

    bool write_diff_edges = true;
    if (write_diff_edges) { // Create edge mesh connecting corresponding points from before and
                            // after
        std::cout << "Creating edge mesh connecting before/after points" << std::endl;
        int num_points = points_before.rows();

        // Combine vertices: first num_points rows are points_before, next num_points rows are
        // points_after
        Eigen::MatrixXd edge_vertices(2 * num_points, 3);
        edge_vertices.topRows(num_points) = points_before;
        edge_vertices.bottomRows(num_points) = points_after;

        // Create edges: each edge connects point i from points_before to point i from points_after
        Eigen::MatrixXi edges(num_points, 2);
        for (int i = 0; i < num_points; i++) {
            edges(i, 0) = i; // Index in points_before
            edges(i, 1) = i + num_points; // Index in points_after
        }

        // Write edge mesh to VTU file
        std::string edge_mesh_filename = "tracking_edges.vtu";
        vtu_utils::write_edge_mesh_to_vtu(edge_vertices, edges, edge_mesh_filename);
        std::cout << "✓ Successfully wrote edge mesh to: " << edge_mesh_filename << std::endl;
    }
}

// Point-tracking specific functions extracted from tet_track_operations.cpp

// handle consolidate point version
void handle_consolidate_tet(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    std::vector<query_point_tet>& query_points,
    bool forward)
{
    std::cout << "Handling Consolidate" << std::endl;
    if (!forward) {
        // backward
        igl::parallel_for(query_points.size(), [&](int id) {
            query_point_tet& qp = query_points[id];
            if (qp.t_id >= 0) {
                if (tet_ids_maps[qp.t_id] != qp.t_id) {
                    qp.t_id = tet_ids_maps[qp.t_id];
                }
                for (int i = 0; i < 4; i++) {
                    if (vertex_ids_maps[qp.tv_ids[i]] != qp.tv_ids[i]) {
                        qp.tv_ids[i] = vertex_ids_maps[qp.tv_ids[i]];
                    }
                }
            }
        });
    } else {
        // forward
        igl::parallel_for(query_points.size(), [&](int id) {
            query_point_tet& qp = query_points[id];
            if (qp.t_id >= 0) {
                auto it = std::find(tet_ids_maps.begin(), tet_ids_maps.end(), qp.t_id);
                if (it != tet_ids_maps.end()) {
                    qp.t_id = std::distance(tet_ids_maps.begin(), it);
                }
                for (int i = 0; i < 4; i++) {
                    auto it_v =
                        std::find(vertex_ids_maps.begin(), vertex_ids_maps.end(), qp.tv_ids[i]);
                    if (it_v != vertex_ids_maps.end()) {
                        qp.tv_ids[i] = std::distance(vertex_ids_maps.begin(), it_v);
                    } else {
                        std::cout << "Error: vertex not found" << std::endl;
                    }
                }
            }
        });
    }
}

void handle_local_mapping_tet(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    std::vector<query_point_tet>& query_points)
{
    std::cout << "Handling Local Mapping" << std::endl;
    for (int id = 0; id < query_points.size(); id++) {
        query_point_tet& qp = query_points[id];
        // TODO: maybe for here is not needed
        if (qp.t_id < 0) continue;
        auto it = std::find(id_map_after.begin(), id_map_after.end(), qp.t_id);
        if (it == id_map_after.end()) continue; // not found

        int local_index_in_t_after = std::distance(id_map_after.begin(), it);
        std::cout << "Input barycentric coordinates: ";
        for (int i = 0; i < qp.bc.size(); ++i) {
            std::cout << std::setprecision(16) << qp.bc(i);
            if (i < qp.bc.size() - 1) std::cout << " ";
        }
        std::cout << std::endl;


        // get position here
        Eigen::Vector3d p(0, 0, 0);
        for (int i = 0; i < 4; i++) {
            int v_id = qp.tv_ids[i];
            auto it_v = std::find(v_id_map_after.begin(), v_id_map_after.end(), v_id);
            if (it_v == v_id_map_after.end()) {
                std::cout << "Error: vertex not found" << std::endl;
                continue;
            }

            int local_index_in_v_after = std::distance(v_id_map_after.begin(), it_v);
            p += V_after.row(local_index_in_v_after) * qp.bc(i);
        }

        // compute bc of the p in (V, T)_before
        // First try orient3d version
        auto result = findTetContainingPointOrient3d(V_before, T_before, p);
        auto [t_id_before, bc_before] = result;

        if (t_id_before == -1) {
            // Orient3d failed, try rational version
            std::cout << "Orient3d failed, trying rational version for point (" << p.transpose()
                      << ")" << std::endl;

            // Recompute position using rational arithmetic
            Eigen::Matrix<wmtk::Rational, 3, 1> p_rational =
                Eigen::Matrix<wmtk::Rational, 3, 1>::Zero();
            Eigen::Matrix<wmtk::Rational, 4, 3> tet_vertices_rational;
            Eigen::Matrix<wmtk::Rational, 4, 1> bc_rational;

            // Convert barycentric coordinates to rational
            for (int i = 0; i < 4; ++i) {
                bc_rational(i) = wmtk::Rational(qp.bc(i));
            }

            // Get tetrahedron vertices and convert to rational
            for (int i = 0; i < 4; i++) {
                int v_id = qp.tv_ids[i];
                auto it_v = std::find(v_id_map_after.begin(), v_id_map_after.end(), v_id);
                if (it_v == v_id_map_after.end()) {
                    std::cout << "Error: vertex not found in rational computation" << std::endl;
                    continue;
                }

                int local_index_in_v_after = std::distance(v_id_map_after.begin(), it_v);
                // Convert vertex coordinates to rational
                for (int j = 0; j < 3; ++j) {
                    tet_vertices_rational(i, j) =
                        wmtk::Rational(V_after(local_index_in_v_after, j));
                }
            }

            // Compute position using rational arithmetic
            p_rational = barycentricToWorldRational(bc_rational, tet_vertices_rational);

            // Convert matrices to rational for findTetContainingPointRational
            auto V_before_rational = toRationalMatrix(V_before);

            // Call rational version
            auto rational_result =
                findTetContainingPointRational(V_before_rational, T_before, p_rational);

            if (rational_result.first != -1) {
                t_id_before = rational_result.first;
                bc_before = toDoubleBarycentric(rational_result.second);
                std::cout << "Rational version succeeded: found tetrahedron " << t_id_before
                          << " with barycentric coordinates " << bc_before.transpose() << std::endl;
            } else {
                std::cout << "Error: Both orient3d and rational versions failed for point"
                          << std::endl;
                std::cout
                    << "Finding closest valid tetrahedron and clipping barycentric coordinates..."
                    << std::endl;

                // Find the tetrahedron with the "most valid" barycentric coordinates
                // (least negative or least exceeding 1)
                int best_tet_id = -1;
                Eigen::Matrix<wmtk::Rational, 4, 1> best_bc_rational;
                double best_score = -1e10; // Higher score is better

                auto V_before_rational = toRationalMatrix(V_before);

                for (int tet_idx = 0; tet_idx < T_before.rows(); ++tet_idx) {
                    // Extract vertices for this tetrahedron
                    Eigen::Matrix<wmtk::Rational, 3, 1> v0 =
                        V_before_rational.row(T_before(tet_idx, 0));
                    Eigen::Matrix<wmtk::Rational, 3, 1> v1 =
                        V_before_rational.row(T_before(tet_idx, 1));
                    Eigen::Matrix<wmtk::Rational, 3, 1> v2 =
                        V_before_rational.row(T_before(tet_idx, 2));
                    Eigen::Matrix<wmtk::Rational, 3, 1> v3 =
                        V_before_rational.row(T_before(tet_idx, 3));

                    // Construct matrix for barycentric coordinate calculation
                    Eigen::Matrix<wmtk::Rational, 4, 4> M;
                    M << v0(0), v1(0), v2(0), v3(0), v0(1), v1(1), v2(1), v3(1), v0(2), v1(2),
                        v2(2), v3(2), wmtk::Rational(1), wmtk::Rational(1), wmtk::Rational(1),
                        wmtk::Rational(1);

                    Eigen::Matrix<wmtk::Rational, 4, 1> rhs;
                    rhs << p_rational(0), p_rational(1), p_rational(2), wmtk::Rational(1);

                    // Solve using Cramer's rule
                    Eigen::Matrix<wmtk::Rational, 4, 1> bc_tet;
                    wmtk::Rational det = M.determinant();
                    if (det == wmtk::Rational(0)) {
                        continue; // Skip degenerate tetrahedron
                    }

                    for (int j = 0; j < 4; ++j) {
                        Eigen::Matrix<wmtk::Rational, 4, 4> M_j = M;
                        M_j.col(j) = rhs;
                        bc_tet(j) = M_j.determinant() / det;
                    }

                    // Compute score based on how "valid" the barycentric coordinates are
                    // Score is minimum of all coordinates (higher is better)
                    // But also penalize coordinates > 1
                    double score = 1e10;
                    for (int j = 0; j < 4; ++j) {
                        double bc_val = bc_tet(j).to_double();
                        if (bc_val > 1.0) {
                            score = std::min(score, 1.0 - (bc_val - 1.0)); // Penalize overshooting
                        } else {
                            score = std::min(score, bc_val); // Reward positive values
                        }
                    }

                    if (score > best_score) {
                        best_score = score;
                        best_tet_id = tet_idx;
                        best_bc_rational = bc_tet;
                    }
                }

                if (best_tet_id != -1) {
                    // Clip barycentric coordinates to valid range [0, 1]
                    for (int j = 0; j < 4; ++j) {
                        wmtk::Rational bc_val = best_bc_rational(j);
                        if (bc_val < wmtk::Rational(0)) {
                            best_bc_rational(j) = wmtk::Rational(0);
                        } else if (bc_val > wmtk::Rational(1)) {
                            best_bc_rational(j) = wmtk::Rational(1);
                        }
                    }

                    // Renormalize to ensure sum equals 1
                    wmtk::Rational sum = wmtk::Rational(0);
                    for (int j = 0; j < 4; ++j) {
                        sum += best_bc_rational(j);
                    }
                    if (sum > wmtk::Rational(0)) {
                        for (int j = 0; j < 4; ++j) {
                            best_bc_rational(j) = best_bc_rational(j) / sum;
                        }
                    }

                    t_id_before = best_tet_id;
                    bc_before = toDoubleBarycentric(best_bc_rational);

                    std::cout << "Found closest tetrahedron " << best_tet_id
                              << " with clipped barycentric coordinates " << bc_before.transpose()
                              << " (score: " << best_score << ")" << std::endl;
                } else {
                    std::cout << "Error: Could not find any valid tetrahedron approximation"
                              << std::endl;
                    continue;
                }
            }
        }

        if (t_id_before == -1) {
            std::cout << "Error: Point not in T_before" << std::endl;
            continue;
        }

        // write out the change
        std::cout << "Change: " << qp.t_id << "->" << id_map_before[t_id_before] << std::endl;
        std::cout << "BC:" << qp.bc.transpose() << "->" << bc_before.transpose() << std::endl;

        // update the query point
        qp.t_id = id_map_before[t_id_before];
        for (int i = 0; i < 4; i++) {
            qp.tv_ids[i] = v_id_map_before[T_before(t_id_before, i)];
            qp.bc(i) = std::max(0.0, std::min(1.0, bc_before(i)));
        }
        qp.bc /= qp.bc.sum(); // normalize
    }
}

void track_point_one_operation_tet(
    const json& operation_log,
    std::vector<query_point_tet>& query_points,
    bool do_forward,
    bool use_rational,
    int operation_id)
{
    std::string operation_name;
    operation_name = operation_log["operation_name"];

    if (operation_name == "MeshConsolidate") {
        std::cout << "This Operations is Consolidate" << std::endl;
        std::vector<int64_t> tet_ids_maps;
        std::vector<int64_t> vertex_ids_maps;
        parse_consolidate_file_tet(operation_log, tet_ids_maps, vertex_ids_maps);

        handle_consolidate_tet(tet_ids_maps, vertex_ids_maps, query_points, do_forward);
    } else {
        std::cout << "This Operations is " << operation_name << std::endl;
        Eigen::MatrixXi T_after, T_before;
        Eigen::MatrixXd V_after, V_before;
        std::vector<int64_t> id_map_after, id_map_before;
        std::vector<int64_t> v_id_map_after, v_id_map_before;
        parse_non_collapse_file_tet(
            operation_log,
            V_before,
            T_before,
            id_map_before,
            v_id_map_before,
            V_after,
            T_after,
            id_map_after,
            v_id_map_after,
            operation_id);

        if (do_forward) {
            handle_local_mapping_tet(
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                query_points);
        } else {
            handle_local_mapping_tet(
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                query_points);
        }
    }
}

void track_point_tet(
    const std::filesystem::path& dirPath,
    std::vector<query_point_tet>& query_points,
    bool do_forward,
    bool use_rational)
{
    BatchOperationLogReader reader(dirPath);
    size_t total_ops = reader.get_total_operations();

    if (total_ops == 0) {
        std::cerr << "No operation logs found in " << dirPath << std::endl;
        return;
    }

    std::cout << "Found " << total_ops << " operations in "
              << (reader.is_batch_format() ? "batch" : "legacy") << " format" << std::endl;

    for (size_t i = 0; i < total_ops; ++i) {
        size_t operation_index = i;
        if (!do_forward) {
            operation_index = total_ops - 1 - i;
        }

        json operation_log = reader.get_operation(operation_index);
        if (operation_log.empty()) {
            std::cerr << "Failed to read operation " << operation_index << std::endl;
            continue;
        }

        std::cout << "Trace Operations number: " << operation_index << std::endl;
        track_point_one_operation_tet(
            operation_log,
            query_points,
            do_forward,
            use_rational,
            static_cast<int>(operation_index));
    }
}

} // namespace tet_point_tracking

#include "tet_point_tracking_app.hpp"
#include <iostream>
#include <unordered_map>
#include <wmtk/utils/Rational.hpp>
#include "tet_point_tracking.hpp"
#include "tet_track_operations_internal.hpp"
#include "vtu_utils.hpp"

namespace tet_point_tracking {

using tet_tracking_utils::to_double_scalar;

////////////////////////////////////////////////////////////
// Application Functions
////////////////////////////////////////////////////////////
template <typename CoordType>
Eigen::MatrixXd write_points_to_file(
    const std::vector<query_point_tet_t<CoordType>>& query_points,
    const Eigen::MatrixXd& V,
    const std::string& filename)
{
    // Compute point coordinates and store in Eigen::MatrixXd
    Eigen::MatrixXd point_coords(query_points.size(), 3);

    for (int i = 0; i < query_points.size(); i++) {
        const auto& qp = query_points[i];
        Eigen::Vector3d p(0, 0, 0);
        for (int j = 0; j < 4; j++) {
            p += to_double_scalar(qp.bc(j)) * V.row(qp.tv_ids[j]).transpose();
        }
        point_coords.row(i) = p;
    }

    // Write to VTU file using vtu_utils
    vtu_utils::write_point_mesh_to_vtu(point_coords, filename);

    return point_coords;
}


// Sample points on boundary tetrahedrons (internal function)
static std::vector<query_point_tet> sample_boundary_tet_points(const Eigen::MatrixXi& T_after)
{
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

    return boundary_query_points;
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

    // Sample some points on boundary tetrahedrons
    std::vector<query_point_tet> query_points = sample_boundary_tet_points(T_after);

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

// Explicit instantiations
template Eigen::MatrixXd write_points_to_file<double>(
    const std::vector<query_point_tet_t<double>>&,
    const Eigen::MatrixXd&,
    const std::string&);
template Eigen::MatrixXd write_points_to_file<wmtk::Rational>(
    const std::vector<query_point_tet_t<wmtk::Rational>>&,
    const Eigen::MatrixXd&,
    const std::string&);

} // namespace tet_point_tracking

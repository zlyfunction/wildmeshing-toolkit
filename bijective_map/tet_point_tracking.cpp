#include "tet_point_tracking.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "track_operations_tet.hpp"

namespace tet_point_tracking {

void write_points_to_file(
    const std::vector<query_point_tet>& query_points,
    const Eigen::MatrixXd& V,
    const std::string& filename)
{
    std::ofstream file(filename);
    for (int i = 0; i < query_points.size(); i++) {
        auto& qp = query_points[i];
        Eigen::Vector3d p(0, 0, 0);
        for (int j = 0; j < 4; j++) {
            p += qp.bc(j) * V.row(qp.tv_ids[j]);
        }
        file << p[0] << "," << p[1] << ", " << p[2] << "\n";
    }
    file.close();
}

void run_back_tracking(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir)
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
            std::string face_key = std::to_string(face[0]) + "_" + std::to_string(face[1]) +
                                   "_" + std::to_string(face[2]);
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
            std::string face_key = std::to_string(face[0]) + "_" + std::to_string(face[1]) +
                                   "_" + std::to_string(face[2]);
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
    write_points_to_file(query_points, V_after, "points_after_remesh.csv");
    track_point_tet(operation_logs_dir, query_points, false, false);

    std::cout << "Writing points to file after back tracking" << std::endl;
    write_points_to_file(query_points, V_before, "points_after_back_tracking.csv");
}

} // namespace tet_point_tracking

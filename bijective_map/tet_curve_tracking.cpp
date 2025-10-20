#include "tet_curve_tracking.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>
#include <igl/tet_tet_adjacency.h>
#include "track_operations_tet.hpp"

namespace tet_curve_tracking {

void write_curve_points_to_file(
    const Eigen::MatrixXd& V,
    const query_curve_tet& curve,
    const std::string& filename1,
    const std::string& filename2)
{
    // Convert barycentric coordinates to real positions and write to CSV files
    std::ofstream start_points_file(filename1);
    std::ofstream end_points_file(filename2);

    for (const auto& seg : curve.segments) {
        // Get vertices of the tetrahedron
        Eigen::Vector4i tet_verts = seg.tv_ids;

        // Calculate real position for start point (bcs[0])
        Eigen::Vector3d start_pos = Eigen::Vector3d::Zero();
        for (int i = 0; i < 4; i++) {
            start_pos += seg.bcs[0](i) * V.row(tet_verts(i)).transpose();
        }

        // Calculate real position for end point (bcs[1])
        Eigen::Vector3d end_pos = Eigen::Vector3d::Zero();
        for (int i = 0; i < 4; i++) {
            end_pos += seg.bcs[1](i) * V.row(tet_verts(i)).transpose();
        }

        // Write to CSV files
        start_points_file << start_pos(0) << "," << start_pos(1) << "," << start_pos(2) << "\n";
        end_points_file << end_pos(0) << "," << end_pos(1) << "," << end_pos(2) << "\n";
    }

    start_points_file.close();
    end_points_file.close();

    std::cout << "Written curve points to " << filename1 << " and " << filename2 << std::endl;
}

void run_back_tracking_curve(
    const Eigen::MatrixXi& T_after,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXd& V_before,
    const std::filesystem::path& operation_logs_dir)
{
    // 1. sample a curve in the tet mesh (output mesh)
    Eigen::MatrixXi TT, TTi;
    igl::tet_tet_adjacency(T_after, TT, TTi);
    query_curve_tet curve;
    int curve_length = 10;

    auto mid_point_on_face = [](int i) {
        if (i == 0) {
            // face [0,1,2]
            return Eigen::Vector4d(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        } else if (i == 1) {
            // face [0,1,3]
            return Eigen::Vector4d(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0);
        } else if (i == 2) {
            // face [1,2,3]
            return Eigen::Vector4d(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        } else if (i == 3) {
            // face [2,0,3]
            return Eigen::Vector4d(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0);
        } else {
            throw std::runtime_error("Invalid face index");
        }
    };

    std::cout << "Start sampling the curve" << std::endl;
    // Set random seed for reproducibility
    srand(42);
    // sample the start point
    int start_tet_id = rand() % T_after.rows();
    std::unordered_set<int> visited_tets;

    int current_tet_id = start_tet_id;
    int prev_face = -1;

    for (int seg_id = 0; seg_id < curve_length; seg_id++) {
        if (current_tet_id == -1 || visited_tets.count(current_tet_id) > 0) {
            break;
        }
        visited_tets.insert(current_tet_id);
        int first_face = prev_face == -1 ? rand() % 4 : prev_face;
        int second_face = (first_face + 1 + rand() % 3) % 4;

        query_segment_tet seg;
        seg.t_id = current_tet_id;
        seg.bcs[0] = mid_point_on_face(first_face);
        seg.bcs[1] = mid_point_on_face(second_face);
        seg.tv_ids = T_after.row(current_tet_id);
        curve.segments.push_back(seg);

        std::cout << "TTi.row(current_tet_id): " << TTi.row(current_tet_id) << std::endl;
        std::cout << "TT.row(current_tet_id): " << TT.row(current_tet_id) << std::endl;
        prev_face = TTi(current_tet_id, second_face);
        current_tet_id = TT(current_tet_id, second_face);
    }

    curve.next_segment_ids.resize(curve.segments.size());
    for (int i = 0; i < curve.segments.size() - 1; i++) {
        curve.next_segment_ids[i] = i + 1;
    }
    curve.next_segment_ids[curve.segments.size() - 1] = -1;

    std::cout << "curve.segments.size(): " << curve.segments.size() << std::endl;

    write_curve_points_to_file(V_after, curve, "curve_start_points.csv", "curve_end_points.csv");

    // 2. back track the curve
    track_curve_tet(operation_logs_dir, curve, false, false);

    // 3. write the curve to a file
    write_curve_points_to_file(V_before, curve, "curve_start_points_before.csv", "curve_end_points_before.csv");
}

} // namespace tet_curve_tracking

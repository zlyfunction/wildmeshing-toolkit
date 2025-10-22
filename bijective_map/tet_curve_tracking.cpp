#include "tet_curve_tracking.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>
#include <igl/tet_tet_adjacency.h>
#include "tet_track_operations.hpp"
#include "tet_track_operations_internal.hpp"
#include "tet_point_tracking.hpp"
#include "batch_operation_log_reader.hpp"

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

// handle consolidate curve version
void handle_consolidate_tet_curve(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    query_curve_tet& curve,
    bool forward)
{
    std::cout << "Handling Consolidatefor curve" << std::endl;
    if (!forward) {
        // backward
        // TODO: maybe use igl::parallel_for
        for (int id = 0; id < curve.segments.size(); id++) {
            auto& qs = curve.segments[id];
            if (qs.t_id >= 0) {
                qs.t_id = tet_ids_maps[qs.t_id];
            }
            for (int j = 0; j < 4; j++) {
                qs.tv_ids[j] = vertex_ids_maps[qs.tv_ids[j]];
            }
        }
    } else {
        // forward
        for (int id = 0; id < curve.segments.size(); id++) {
            auto& qs = curve.segments[id];
            if (qs.t_id >= 0) {
                auto it = std::find(tet_ids_maps.begin(), tet_ids_maps.end(), qs.t_id);
                if (it != tet_ids_maps.end()) {
                    qs.t_id = std::distance(tet_ids_maps.begin(), it);
                }
                for (int j = 0; j < 4; j++) {
                    auto it_v =
                        std::find(vertex_ids_maps.begin(), vertex_ids_maps.end(), qs.tv_ids[j]);
                    if (it_v != vertex_ids_maps.end()) {
                        qs.tv_ids[j] = std::distance(vertex_ids_maps.begin(), it_v);
                    } else {
                        std::cout << "Error: vertex not found" << std::endl;
                    }
                }
            }
        }
    }
}

void handle_one_segment_tet(
    query_curve_tet& curve,
    int id,
    std::vector<query_point_tet>& current_qps,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T,
    const std::vector<int64_t>& id_map,
    const std::vector<int64_t>& v_id_map,
    Eigen::MatrixXi& TT,
    Eigen::MatrixXi& TTi,
    double eps)
{
    auto& seg = curve.segments[id];
    auto& p0 = current_qps[0];
    auto& p1 = current_qps[1];

    // TODO: get the exact version of this function
    auto is_same_tet = [&](query_point_tet& qp0, query_point_tet& qp1) {
        return qp0.t_id == qp1.t_id;
    };


    // libigl face ordering → vertex‑to‑face map for TT access
    // if bc[i] == 0  then v2f[i] is the face
    constexpr int v2f[4] = {2, 3, 1, 0};


    auto tet_vertices = [&](int64_t tid) {
        std::cout << "V.rows(): " << V.rows() << std::endl;
        std::cout << "tid = " << tid << std::endl;
        std::cout << "T.rows(): " << T.rows() << std::endl;

        Eigen::Matrix<double, 4, 3> v;
        v.row(0) = V.row(T(tid, 0));
        v.row(1) = V.row(T(tid, 1));
        v.row(2) = V.row(T(tid, 2));
        v.row(3) = V.row(T(tid, 3));
        return v;
    };


    if (TT.rows() == 0) {
        std::cout << "Getting TT and TTi" << std::endl;
        igl::tet_tet_adjacency(T, TT, TTi);
    }

    std::cout << "Getting p1_local_tid" << std::endl;
    int p1_local_tid = -1;
    {
        std::cout << "p1.t_id: " << p1.t_id << std::endl;
        std::cout << "id_map size: " << id_map.size() << std::endl;
        std::cout << "id_map contents: ";
        for (size_t i = 0; i < id_map.size(); ++i) {
            std::cout << id_map[i] << " ";
        }
        std::cout << std::endl;
        auto it = std::find(id_map.begin(), id_map.end(), p1.t_id);
        p1_local_tid = std::distance(id_map.begin(), it);
    }

    std::cout << "Getting p1_world" << std::endl;
    auto p1_tet_vertices = tet_vertices(p1_local_tid);
    const Eigen::Vector3d p1_world = barycentric_to_world_tet(p1.bc, p1_tet_vertices);

    if (is_same_tet(p0, p1)) {
        seg.t_id = p0.t_id;
        seg.bcs[0] = p0.bc;
        seg.bcs[1] = p1.bc;
        seg.tv_ids = p0.tv_ids;
        return;
    } else {
        std::cout << "Start splitting: " << std::endl;
        int old_next_seg = curve.next_segment_ids[id];
        auto it = std::find(id_map.begin(), id_map.end(), current_qps[0].t_id);
        int current_local_tid = std::distance(id_map.begin(), it);

        query_point_tet cur = p0;

        while (true) {
            const auto v_cur = tet_vertices(current_local_tid);
            const Eigen::Vector4d b0 = cur.bc;
            Eigen::Vector4d b1;

            // If p1 in same tet, use its barycentrics directly; else compute
            if (is_same_tet(cur, p1)) {
                query_segment_tet new_seg{cur.t_id, {b0, p1.bc}, p1.tv_ids};
                curve.segments.push_back(new_seg);
                curve.next_segment_ids[id] = curve.segments.size() - 1;
                curve.next_segment_ids.push_back(old_next_seg);
                break;
            } else {
                b1 = world_to_barycentric_tet(p1_world, v_cur);
            }

            std::cout << "b0: " << b0.transpose() << std::endl;
            std::cout << "b1: " << b1.transpose() << std::endl;

            double t_exit = 1.0;
            int exit_fid = -1;

            for (int i = 0; i < 4; i++) {
                std::cout << "i: " << i << std::endl;
                double denom = b1[i] - b0[i];
                if (abs(denom) < eps) {
                    std::cout << "ray parallel to the face" << std::endl;
                    continue;
                }
                double t = b0[i] / (b0[i] - b1[i]);
                std::cout << "t: " << t << std::endl;

                if (t < eps) continue; // on the wrong side

                Eigen::Vector4d bc_intersect = b0 + t * (b1 - b0);
                std::cout << "bc_intersect: " << bc_intersect.transpose() << std::endl;

                // check if all bc_intersect are non-negative and less than 1
                if ((bc_intersect.array() >= -eps).all() &&
                    (bc_intersect.array() <= 1.0 + eps).all()) {
                    t_exit = t;
                    exit_fid = v2f[i];
                    break;
                }
            }

            // TODO: for debug
            break;
        } // end of while
        // If p1 is inside current tet → finish
    } // end of else
}


void handle_local_mapping_tet_curve(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_curve_tet& curve)
{
    int curve_length = curve.segments.size();
    double eps = 1e-8;
    std::cout << "Handling Local Mapping for curve" << std::endl;
    Eigen::MatrixXi TT, TTi; // connectivity of T_before

    // TODO: can we do parallel here?
    for (int id = 0; id < curve_length; id++) {
        auto& seg = curve.segments[id];
        query_point_tet qp0 = {seg.t_id, seg.bcs[0], seg.tv_ids};
        query_point_tet qp1 = {seg.t_id, seg.bcs[1], seg.tv_ids};
        std::vector<query_point_tet> qps = {qp0, qp1};

        std::cout << "Handling local mapping for segment " << id << std::endl;
        tet_point_tracking::handle_local_mapping_tet(
            V_before,
            T_before,
            id_map_before,
            v_id_map_before,
            V_after,
            T_after,
            id_map_after,
            v_id_map_after,
            qps);
        std::cout << "Handling one segment, get intersections." << std::endl;
        // TODO: implement this function
        handle_one_segment_tet(
            curve,
            id,
            qps,
            V_before,
            T_before,
            id_map_before,
            v_id_map_before,
            TT,
            TTi,
            eps);
    }
}

void track_curve_one_operation_tet(
    const json& operation_log,
    query_curve_tet& curve,
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

        handle_consolidate_tet_curve(tet_ids_maps, vertex_ids_maps, curve, do_forward);
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
            handle_local_mapping_tet_curve(
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                curve);
        } else {
            handle_local_mapping_tet_curve(
                V_before,
                T_before,
                id_map_before,
                v_id_map_before,
                V_after,
                T_after,
                id_map_after,
                v_id_map_after,
                curve);
        }
    }
}

void track_curve_tet(
    const std::filesystem::path& dirPath,
    query_curve_tet& curve,
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
        track_curve_one_operation_tet(operation_log, curve, do_forward, use_rational, static_cast<int>(operation_index));
    }
}

} // namespace tet_curve_tracking

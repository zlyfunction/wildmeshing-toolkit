#include "track_line_app.hpp"
#include <igl/Timer.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/parallel_for.h>
#include <wmtk/utils/orient.hpp>

void track_line_one_operation(const json& operation_log, query_curve& curve, bool do_forward)
{
    std::string operation_name;
    operation_name = operation_log["operation_name"];

    if (operation_name == "MeshConsolidate") {
        std::cout << "This Operations is Consolidate" << std::endl;
        std::vector<int64_t> face_ids_maps;
        std::vector<int64_t> vertex_ids_maps;
        parse_consolidate_file(operation_log, face_ids_maps, vertex_ids_maps);
        if (do_forward) {
            handle_consolidate_forward(face_ids_maps, vertex_ids_maps, curve);
        } else {
            handle_consolidate(face_ids_maps, vertex_ids_maps, curve);
        }
    } else if (operation_name == "TriEdgeSwap" || operation_name == "AttributesUpdate") {
        std::cout << "This Operations is" << operation_name << std::endl;
        Eigen::MatrixXi F_after, F_before;
        Eigen::MatrixXd V_after, V_before;
        std::vector<int64_t> id_map_after, id_map_before;
        std::vector<int64_t> v_id_map_after, v_id_map_before;
        bool is_skipped;

        parse_non_collapse_file(
            operation_log,
            is_skipped,
            V_before,
            F_before,
            id_map_before,
            v_id_map_before,
            V_after,
            F_after,
            id_map_after,
            v_id_map_after);

        if (is_skipped) {
            return;
        }

        if (do_forward) {
            handle_swap_edge_curve(
                V_after,
                F_after,
                id_map_after,
                v_id_map_after,
                V_before,
                F_before,
                id_map_before,
                v_id_map_before,
                curve);
        } else {
            handle_swap_edge_curve(
                V_before,
                F_before,
                id_map_before,
                v_id_map_before,
                V_after,
                F_after,
                id_map_after,
                v_id_map_after,
                curve);
        }
    } else if (operation_name == "EdgeSplit") {
        std::cout << "This Operations is EdgeSplit" << std::endl;
        Eigen::MatrixXi F_after, F_before;
        Eigen::MatrixXd V_after, V_before;
        std::vector<int64_t> id_map_after, id_map_before;
        std::vector<int64_t> v_id_map_after, v_id_map_before;
        bool is_skipped;
        parse_non_collapse_file(
            operation_log,
            is_skipped,
            V_before,
            F_before,
            id_map_before,
            v_id_map_before,
            V_after,
            F_after,
            id_map_after,
            v_id_map_after);
        // TODO:
    } else if (operation_name == "EdgeCollapse") {
        std::cout << "This Operations is EdgeCollapse" << std::endl;
        Eigen::MatrixXi F_after, F_before;
        Eigen::MatrixXd UV_joint;
        std::vector<int64_t> v_id_map_joint;
        std::vector<int64_t> id_map_after, id_map_before;
        parse_edge_collapse_file(
            operation_log,
            UV_joint,
            F_before,
            F_after,
            v_id_map_joint,
            id_map_before,
            id_map_after);

        if (do_forward) {
            handle_collapse_edge_curve(
                UV_joint,
                F_after,
                F_before,
                v_id_map_joint,
                id_map_after,
                id_map_before,
                curve);
        } else {
            handle_collapse_edge_curve(
                UV_joint,
                F_before,
                F_after,
                v_id_map_joint,
                id_map_before,
                id_map_after,
                curve);
        }
    } else {
        // std::cout << "This Operations is not implemented" << std::endl;
    }
#ifdef DEBUG_CURVES
    // save the curve
    std::string curve_file = "./curve_debug/curve_" + std::to_string(file_id) + ".json";
    std::vector<query_curve> curves{curve};
    save_query_curves(curves, curve_file);
#endif
}

void track_line(path dirPath, query_curve& curve, bool do_forward)
{
    namespace fs = std::filesystem;
    int maxIndex = -1;

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.path().filename().string().find("operation_log_") != std::string::npos) {
            ++maxIndex;
        }
    }

    for (int i = maxIndex; i >= 0; --i) {
        int file_id = i;
        if (do_forward) {
            file_id = maxIndex - i;
        }
        fs::path filePath = dirPath / ("operation_log_" + std::to_string(file_id) + ".json");
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            continue;
        }
        json operation_log;
        file >> operation_log;

        std::cout << "Trace Operations number: " << file_id << std::endl;

        track_line_one_operation(operation_log, curve, do_forward);

        file.close();
    }
}

void track_lines(path dirPath, std::vector<query_curve>& curves, bool do_forward)
{
    // use igl parallel_for
    igl::parallel_for(curves.size(), [&](int i) { track_line(dirPath, curves[i], do_forward); });
}


#include <igl/triangle_triangle_adjacency.h>
query_curve generate_curve(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const int curve_length,
    const int start_face_id)
{
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);
    std::vector<int> visited_faces(F.rows(), 0);

    int current_face_id = start_face_id;
    int current_edge_id = 0;

    query_curve curve;

    Eigen::Matrix3d bary_coord_ref;
    bary_coord_ref << 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 0, 0.5;

    while (curve.segments.size() < curve_length) {
        visited_faces[current_face_id] = 1;
        int next_edge_id = (current_edge_id + 1) % 3;
        if (TT(current_face_id, next_edge_id) == -1 ||
            visited_faces[TT(current_face_id, next_edge_id)] == 1) {
            next_edge_id = (current_edge_id + 2) % 3;
        }

        if (TT(current_face_id, next_edge_id) == -1 ||
            visited_faces[TT(current_face_id, next_edge_id)] == 1) {
            break;
        }

        std::cout << "current_face_id: " << current_face_id << std::endl;
        std::cout << "current_edge_id: " << current_edge_id << std::endl;
        std::cout << "next_edge_id: " << next_edge_id << std::endl;
        std::cout << "TT.row(current_face_id): " << TT.row(current_face_id) << std::endl;
        std::cout << "TTi.row(current_face_id): " << TTi.row(current_face_id) << std::endl;


        query_segment seg;
        seg.f_id = current_face_id;
        seg.bcs[0] = bary_coord_ref.row(current_edge_id);
        seg.bcs[1] = bary_coord_ref.row(next_edge_id);
        seg.fv_ids = F.row(current_face_id);

        curve.segments.push_back(seg);

        current_edge_id = TTi(current_face_id, next_edge_id);
        current_face_id = TT(current_face_id, next_edge_id);
    }

    curve.next_segment_ids.resize(curve.segments.size());
    for (int i = 0; i < curve.segments.size() - 1; i++) {
        curve.next_segment_ids[i] = i + 1;
    }
    curve.next_segment_ids[curve.segments.size() - 1] = -1;

    return curve;
}

void back_track_one_curve_app(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXi& F_out,
    const path& operation_logs_dir)
{
    query_curve curve_in = generate_curve(V_out, F_out, 10, 0);
    for (const auto& seg : curve_in.segments) {
        std::cout << "f_id: " << seg.f_id << std::endl;
        std::cout << "bcs: " << seg.bcs[0] << ", " << seg.bcs[1] << std::endl;
    }

    query_curve curve = curve_in;
    query_curve curve_origin = curve;
    track_line(operation_logs_dir, curve);

    // render before
    {
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V_out, F_out);
        viewer.data().point_size /= 3;
        for (const auto& seg : curve_origin.segments) {
            Eigen::MatrixXd pts(2, 3);
            for (int i = 0; i < 2; i++) {
                Eigen::Vector3d p(0, 0, 0);
                for (int j = 0; j < 3; j++) {
                    p += V_out.row(seg.fv_ids[j]) * seg.bcs[i](j);
                }
                pts.row(i) = p;
            }
            std::cout << "pts: \n" << pts << std::endl;
            viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
            viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
            viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
        }

        viewer.launch();
    }

    // render after
    {
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V_in, F_in);
        viewer.data().point_size /= 3;
        for (const auto& seg : curve.segments) {
            Eigen::MatrixXd pts(2, 3);
            for (int i = 0; i < 2; i++) {
                Eigen::Vector3d p(0, 0, 0);
                for (int j = 0; j < 3; j++) {
                    p += V_in.row(seg.fv_ids[j]) * seg.bcs[i](j);
                }
                pts.row(i) = p;
            }
            viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
            viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
            viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
        }


        viewer.launch();
    }
}

#include "generate_iso_line.hpp"
void forward_track_iso_lines_app(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    const Eigen::MatrixXd& uv_in,
    const Eigen::MatrixXi& Fuv_in,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXi& F_out,
    const path& operation_logs_dir,
    int N)
{
    // get all curves
    std::vector<query_curve> curves;
    {
        int N = 5;

        auto curve_from_intersections = [&](const std::vector<Intersection>& input_intersections) {
            query_curve curve;
            for (int i = 0; i < input_intersections.size(); i += 2) {
                query_segment seg;
                if (input_intersections[i].fid != input_intersections[i + 1].fid) {
                    std::cout << "something wrong with input_intersections" << std::endl;
                }
                seg.f_id = input_intersections[i].fid;
                seg.bcs[0] = input_intersections[i].barycentric;
                seg.bcs[1] = input_intersections[i + 1].barycentric;
                seg.fv_ids = F_in.row(input_intersections[i].fid);
                curve.segments.push_back(seg);
            }

            curve.next_segment_ids.resize(curve.segments.size());
            for (int i = 0; i < curve.segments.size() - 1; i++) {
                curve.next_segment_ids[i] = i + 1;
            }
            curve.next_segment_ids[curve.segments.size() - 1] = -1;

            return curve;
        };

        for (int k = 0; k < N - 1; k++) {
            double value = 1.0 / N * (k + 1);
            auto intersectionsX = computeIsoLineIntersectionsX(uv_in, Fuv_in, value);
            auto curveX = curve_from_intersections(intersectionsX);
            curves.push_back(curveX);
            auto intersectionsY = computeIsoLineIntersectionsY(uv_in, Fuv_in, value);
            auto curveY = curve_from_intersections(intersectionsY);
            curves.push_back(curveY);
        }
    }

    auto curves_origin = curves;

    {
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V_in, F_in);
        viewer.data().point_size /= 3;
        for (const auto curve_origin : curves_origin) {
            for (const auto& seg : curve_origin.segments) {
                Eigen::MatrixXd pts(2, 3);
                for (int i = 0; i < 2; i++) {
                    Eigen::Vector3d p(0, 0, 0);
                    for (int j = 0; j < 3; j++) {
                        p += V_in.row(seg.fv_ids[j]) * seg.bcs[i](j);
                    }
                    pts.row(i) = p;
                }
                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
            }
        }
        viewer.launch();
    }

    save_query_curves(curves, "curves.in");


    track_lines(operation_logs_dir, curves);


    save_query_curves(curves, "curves.out");

    {
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V_out, F_out);
        viewer.data().point_size /= 3;
        for (const auto& curve : curves) {
            for (const auto& seg : curve.segments) {
                Eigen::MatrixXd pts(2, 3);
                for (int i = 0; i < 2; i++) {
                    Eigen::Vector3d p(0, 0, 0);
                    for (int j = 0; j < 3; j++) {
                        p += V_out.row(seg.fv_ids[j]) * seg.bcs[i](j);
                    }
                    pts.row(i) = p;
                }
                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
            }
        }
        viewer.launch();
    }
}

void check_iso_lines(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXi& F_out,
    const std::vector<query_curve>& curves_in,
    const std::vector<query_curve>& curves_out,
    bool render_before,
    bool render_after)
{
    std::cout << "curves_in sizes:\n";
    for (const auto& c : curves_in) {
        std::cout << c.segments.size() << std::endl;
    }
    std::cout << "curves_out sizes:\n";
    for (const auto& c : curves_out) {
        std::cout << c.segments.size() << std::endl;
    }

    if (render_before) {
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V_in, F_in);
        viewer.data().point_size /= 3;
        for (const auto curve_origin : curves_in) {
            for (const auto& seg : curve_origin.segments) {
                Eigen::MatrixXd pts(2, 3);
                for (int i = 0; i < 2; i++) {
                    Eigen::Vector3d p(0, 0, 0);
                    for (int j = 0; j < 3; j++) {
                        p += V_in.row(seg.fv_ids[j]) * seg.bcs[i](j);
                    }
                    pts.row(i) = p;
                }
                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
            }
        }
        viewer.launch();
    }
    if (render_after) {
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V_out, F_out);
        viewer.data().point_size /= 3;
        for (int i = 0; i < curves_out.size(); i++) {
            const auto& curve = curves_out[i];
            for (const auto& seg : curve.segments) {
                Eigen::MatrixXd pts(2, 3);
                for (int i = 0; i < 2; i++) {
                    Eigen::Vector3d p(0, 0, 0);
                    for (int j = 0; j < 3; j++) {
                        p += V_out.row(seg.fv_ids[j]) * seg.bcs[i](j);
                    }
                    pts.row(i) = p;
                }
                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
            }
        }
        viewer.launch();
    }


    auto doIntersect = [](const Eigen::RowVector2d& p1,
                          const Eigen::RowVector2d& q1,
                          const Eigen::RowVector2d& p2,
                          const Eigen::RowVector2d& q2) -> bool {
        auto onSegment = [](const Eigen::RowVector2d& p,
                            const Eigen::RowVector2d& q,
                            const Eigen::RowVector2d& r) -> bool {
            return q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
                   q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1]);
        };

        int o1 = wmtk::utils::wmtk_orient2d(p1, q1, p2);
        int o2 = wmtk::utils::wmtk_orient2d(p1, q1, q2);
        int o3 = wmtk::utils::wmtk_orient2d(p2, q2, p1);
        int o4 = wmtk::utils::wmtk_orient2d(p2, q2, q1);


        // General case
        if (o1 != o2 && o3 != o4) return true;

        // Special Cases
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;

        return false; // No intersection
    };
    auto count_curve_intersection = [&](const std::vector<query_curve>& curve) {
        for (int i = 0; i < curve.size(); i++) {
            for (int j = i; j < curve.size(); j++) {
                // for (int j = i; j < i + 1; j++) {
                int intersect_count = 0;
                for (int seg_i = 0; seg_i < curve[i].segments.size(); seg_i++) {
                    for (int seg_j = 0; seg_j < curve[j].segments.size(); seg_j++) {
                        if (i == j &&
                            (seg_i == seg_j || curve[i].next_segment_ids[seg_i] == seg_j ||
                             curve[j].next_segment_ids[seg_j] == seg_i)) {
                            continue;
                        }
                        if (curve[i].segments[seg_i].f_id != curve[j].segments[seg_j].f_id) {
                            continue;
                        }

                        Eigen::RowVector2d p1, q1, p2, q2;
                        p1 = curve[i].segments[seg_i].bcs[0].head(2);
                        q1 = curve[i].segments[seg_i].bcs[1].head(2);
                        p2 = curve[j].segments[seg_j].bcs[0].head(2);
                        q2 = curve[j].segments[seg_j].bcs[1].head(2);

                        // if ((p1 - p2).norm() < 1e-8 || (p1 - q2).norm() < 1e-8 ||
                        //     (q1 - p2).norm() < 1e-8 || (q1 - q2).norm() < 1e-8) {
                        //     continue;
                        // }

                        if (doIntersect(p1, q1, p2, q2)) {
                            intersect_count++;
                            // std::cout << "i = " << i << ", seg_i = " << seg_i
                            //           << ", next[seg_i] = " <<
                            //           curves_out[i].next_segment_ids[seg_i]
                            //           << std::endl;
                            // std::cout << "p1: " << p1 << std::endl;
                            // std::cout << "q1: " << q1 << std::endl;
                            // std::cout << "j = " << j << ", seg_j = " << seg_j
                            //           << ", next[seg_j] = " <<
                            //           curves_out[j].next_segment_ids[seg_j]
                            //           << std::endl;
                            // std::cout << "p2: " << p2 << std::endl;
                            // std::cout << "q2: " << q2 << std::endl;

                            // std::cout << "next[seg_i]: "
                            //           << curves_out[i]
                            //                  .segments[curves_out[i].next_segment_ids[seg_i]]
                            //                  .bcs[0]
                            //                  .head(2)
                            //                  .transpose()
                            //           << ", "
                            //           << curves_out[i]
                            //                  .segments[curves_out[i].next_segment_ids[seg_i]]
                            //                  .bcs[1]
                            //                  .head(2)
                            //                  .transpose()
                            //           << std::endl;
                            // std::cout << "next[seg_j]: "
                            //           << curves_out[j]
                            //                  .segments[curves_out[j].next_segment_ids[seg_j]]
                            //                  .bcs[0]
                            //                  .head(2)
                            //                  .transpose()
                            //           << ", "
                            //           << curves_out[j]
                            //                  .segments[curves_out[j].next_segment_ids[seg_j]]
                            //                  .bcs[1]
                            //                  .head(2)
                            //                  .transpose()

                            //           << std::endl;
                            // std::cout << std::endl;
                        }
                    }
                }
                std::cout << "curve " << i << " and curve " << j << " intersect " << intersect_count
                          << " times" << std::endl;
            }
            std::cout << std::endl;
        }
    };

    std::cout << "count curve_in intersection" << std::endl;
    count_curve_intersection(curves_in);
    std::cout << "count curve_out intersection" << std::endl;
    count_curve_intersection(curves_out);
}

#ifdef DEBUG_CURVES
void check_iso_lines_step_by_step(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    const Eigen::MatrixXd& V_out,
    const Eigen::MatrixXi& F_out,
    const path& operation_logs_dir)
{
    std::vector<query_curve> curves_in = load_query_curves("curves.in");
    namespace fs = std::filesystem;
    int file_id = -1;
    int step_size = 1;
    int view_mode = 0;
    auto key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
        if (key == '1') {
            file_id += step_size;
            std::cout << "file_id: " << file_id << std::endl;
        }

        if (key == '0') {
            file_id -= step_size;
            std::cout << "file_id: " << file_id << std::endl;
        }

        if (key == '=') {
            step_size *= 10;
            std::cout << "step_size: " << step_size << std::endl;
        }

        if (key == '-') {
            if (step_size > 1) step_size /= 10;
            std::cout << "step_size: " << step_size << std::endl;
        }

        if (key == ' ') {
            view_mode = 0;
        }

        if (key == '8') {
            view_mode = 1;
        }

        if (key == '9') {
            view_mode = 2;
        }

        if (key == '6') {
            view_mode = 3;
        }

        if (key == '7') {
            view_mode = 4;
        }

        if (file_id >= 0) {
            if (view_mode == 0) {
                // read obj from file
                Eigen::MatrixXd V_cur, Vt_cur, Vn_cur;
                Eigen::MatrixXi F_cur, Ft_cur, Fn_cur;
                fs::path MeshfilePath = operation_logs_dir / ("VF_all_after_operation_" +
                                                              std::to_string(file_id) + ".obj");
                igl::readOBJ(MeshfilePath.string(), V_cur, Vt_cur, Vn_cur, F_cur, Ft_cur, Fn_cur);

                // read F_flag_after_operation_*.txt
                std::vector<int> F_flag;
                fs::path F_flag_path = operation_logs_dir / ("F_flag_after_operation_" +
                                                             std::to_string(file_id) + ".txt");
                std::ifstream file(F_flag_path.string());
                if (file.is_open()) {
                    int flag;
                    while (file >> flag) {
                        F_flag.push_back(flag);
                    }
                    file.close();
                }

                // read curve from file
                std::vector<query_curve> curves_cur =
                    load_query_curves("curve_debug/curve_" + std::to_string(file_id) + ".json");

                // build new F, clean up the ones in F_cur that F_flag is 0
                Eigen::MatrixXi F_new;
                int count = 0;
                for (int i = 0; i < F_cur.rows(); i++) {
                    if (F_flag[i] == 1) {
                        F_new.conservativeResize(count + 1, 3);
                        F_new.row(count) = F_cur.row(i);
                        count++;
                    }
                }

                viewer.data().clear();
                viewer.data().set_mesh(V_cur, F_new);
                viewer.core().align_camera_center(V_in, F_in);
                // viewer.data().point_size /= 3;

                for (const auto& curve : curves_cur) {
                    for (const auto& seg : curve.segments) {
                        Eigen::MatrixXd pts(2, 3);
                        for (int i = 0; i < 2; i++) {
                            Eigen::Vector3d p(0, 0, 0);
                            for (int j = 0; j < 3; j++) {
                                p += V_cur.row(seg.fv_ids[j]) * seg.bcs[i](j);
                            }
                            pts.row(i) = p;
                        }
                        viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                        viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                        viewer.data().add_edges(
                            pts.row(0),
                            pts.row(1),
                            Eigen::RowVector3d(1, 0, 0));
                    }
                }
            } else {
                fs::path filePath =
                    operation_logs_dir / ("operation_log_" + std::to_string(file_id) + ".json");
                std::ifstream file(filePath);
                json operation_log;
                file >> operation_log;

                std::cout << "Trace Operations number: " << file_id << std::endl;
                std::string operation_name;
                operation_name = operation_log["operation_name"];

                Eigen::MatrixXi F_after, F_before;
                Eigen::MatrixXd V_after, V_before;
                std::vector<int64_t> id_map_after, id_map_before;
                std::vector<int64_t> v_id_map_after, v_id_map_before;

                if (operation_name == "TriEdgeSwap" || operation_name == "AttributesUpdate") {
                    std::cout << "This Operations is" << operation_name << std::endl;
                    bool is_skipped;

                    parse_non_collapse_file(
                        operation_log,
                        is_skipped,
                        V_before,
                        F_before,
                        id_map_before,
                        v_id_map_before,
                        V_after,
                        F_after,
                        id_map_after,
                        v_id_map_after);
                } else if (operation_name == "EdgeCollapse") {
                    std::cout << "This Operations is EdgeCollapse" << std::endl;
                    Eigen::MatrixXd UV_joint;
                    std::vector<int64_t> v_id_map_joint;

                    parse_edge_collapse_file(
                        operation_log,
                        UV_joint,
                        F_before,
                        F_after,
                        v_id_map_joint,
                        id_map_before,
                        id_map_after);

                    V_before = UV_joint;
                    V_after = UV_joint;
                    v_id_map_before = v_id_map_joint;
                    v_id_map_after = v_id_map_joint;
                }

                viewer.data().clear();
                if (view_mode == 1) {
                    viewer.data().set_mesh(V_before, F_before);
                    viewer.core().align_camera_center(V_before, F_before);
                    // read curve and draw
                    if (file_id > 0) {
                        std::vector<query_curve> curves_cur = load_query_curves(
                            "curve_debug/curve_" + std::to_string(file_id - 1) + ".json");
                        for (const auto& curve : curves_cur) {
                            for (const auto& seg : curve.segments) {
                                if (std::find(
                                        id_map_before.begin(),
                                        id_map_before.end(),
                                        seg.f_id) == id_map_before.end()) {
                                    continue;
                                }
                                Eigen::MatrixXd pts(2, 2);
                                for (int i = 0; i < 2; i++) {
                                    Eigen::Vector2d p(0, 0);
                                    for (int j = 0; j < 3; j++) {
                                        if (std::find(
                                                v_id_map_before.begin(),
                                                v_id_map_before.end(),
                                                seg.fv_ids[j]) == v_id_map_before.end()) {
                                            std::cout << "not found" << std::endl;
                                        }
                                        int id = std::distance(
                                            v_id_map_before.begin(),
                                            std::find(
                                                v_id_map_before.begin(),
                                                v_id_map_before.end(),
                                                seg.fv_ids[j]));

                                        if (seg.bcs[i](j) < 0 || seg.bcs[i](j) > 1) {
                                            std::cout << "Error: bcs out of range" << std::endl;
                                            std::cout << "seg.bcs[i](j): " << seg.bcs[i](j)
                                                      << std::endl;
                                        }
                                        p += V_before.row(id) * seg.bcs[i](j);
                                    }
                                    pts.row(i) = p;
                                }
                                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                                viewer.data().add_edges(
                                    pts.row(0),
                                    pts.row(1),
                                    Eigen::RowVector3d(1, 0, 0));
                            }
                        }
                    }
                } else if (view_mode == 2) {
                    viewer.data().set_mesh(V_after, F_after);
                    viewer.core().align_camera_center(V_before, F_before);
                    // read curve and draw
                    std::vector<query_curve> curves_cur =
                        load_query_curves("curve_debug/curve_" + std::to_string(file_id) + ".json");
                    for (const auto& curve : curves_cur) {
                        for (const auto& seg : curve.segments) {
                            if (std::find(id_map_after.begin(), id_map_after.end(), seg.f_id) ==
                                id_map_after.end()) {
                                continue;
                            }
                            Eigen::MatrixXd pts(2, 2);
                            for (int i = 0; i < 2; i++) {
                                Eigen::Vector2d p(0, 0);
                                for (int j = 0; j < 3; j++) {
                                    if (std::find(
                                            v_id_map_after.begin(),
                                            v_id_map_after.end(),
                                            seg.fv_ids[j]) == v_id_map_after.end()) {
                                        std::cout << "not found" << std::endl;
                                    }
                                    int offset_for_collapse = 0;
                                    if (operation_name == "EdgeCollapse") {
                                        offset_for_collapse = 1;
                                    }
                                    int id = std::distance(
                                        v_id_map_after.begin(),
                                        std::find(
                                            v_id_map_after.begin() + offset_for_collapse,
                                            v_id_map_after.end(),
                                            seg.fv_ids[j]));
                                    if (seg.bcs[i](j) < 0 || seg.bcs[i](j) > 1) {
                                        std::cout << "Error: bcs out of range" << std::endl;
                                        std::cout << "seg.bcs[i](j): " << seg.bcs[i](j)
                                                  << std::endl;
                                    }

                                    p += V_after.row(id) * seg.bcs[i](j);
                                }
                                pts.row(i) = p;
                            }
                            viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                            viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                            viewer.data().add_edges(
                                pts.row(0),
                                pts.row(1),
                                Eigen::RowVector3d(1, 0, 0));
                        }
                    }
                } else if (view_mode == 3) {
                    Eigen::MatrixXd V_cur, Vt_cur, Vn_cur;
                    Eigen::MatrixXi F_cur, Ft_cur, Fn_cur;
                    fs::path MeshfilePath =
                        operation_logs_dir /
                        ("VF_all_after_operation_" + std::to_string(file_id - 1) + ".obj");
                    igl::readOBJ(
                        MeshfilePath.string(),
                        V_cur,
                        Vt_cur,
                        Vn_cur,
                        F_cur,
                        Ft_cur,
                        Fn_cur);
                    Eigen::MatrixXi F_slice(id_map_before.size(), 3);
                    for (int i = 0; i < id_map_before.size(); i++) {
                        F_slice.row(i) = F_cur.row(id_map_before[i]);
                    }

                    viewer.data().set_mesh(V_cur, F_slice);
                    viewer.core().align_camera_center(V_cur, F_slice);
                    // read curve and draw
                    if (file_id > 0) {
                        std::vector<query_curve> curves_cur = load_query_curves(
                            "curve_debug/curve_" + std::to_string(file_id - 1) + ".json");
                        for (const auto& curve : curves_cur) {
                            for (const auto& seg : curve.segments) {
                                if (std::find(
                                        id_map_before.begin(),
                                        id_map_before.end(),
                                        seg.f_id) == id_map_before.end()) {
                                    continue;
                                }
                                Eigen::MatrixXd pts(2, 3);
                                for (int i = 0; i < 2; i++) {
                                    Eigen::Vector3d p(0, 0, 0);
                                    for (int j = 0; j < 3; j++) {
                                        p += V_cur.row(seg.fv_ids[j]) * seg.bcs[i](j);
                                    }
                                    pts.row(i) = p;
                                }
                                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                                viewer.data().add_edges(
                                    pts.row(0),
                                    pts.row(1),
                                    Eigen::RowVector3d(1, 0, 0));
                            }
                        }
                    }
                } else if (view_mode == 4) {
                    Eigen::MatrixXd V_cur, Vt_cur, Vn_cur;
                    Eigen::MatrixXi F_cur, Ft_cur, Fn_cur;
                    fs::path MeshfilePath = operation_logs_dir / ("VF_all_after_operation_" +
                                                                  std::to_string(file_id) + ".obj");
                    igl::readOBJ(
                        MeshfilePath.string(),
                        V_cur,
                        Vt_cur,
                        Vn_cur,
                        F_cur,
                        Ft_cur,
                        Fn_cur);
                    Eigen::MatrixXi F_slice(id_map_after.size(), 3);
                    for (int i = 0; i < id_map_after.size(); i++) {
                        F_slice.row(i) = F_cur.row(id_map_after[i]);
                    }

                    viewer.data().set_mesh(V_cur, F_slice);
                    viewer.core().align_camera_center(V_cur, F_slice);
                    // read curve and draw
                    if (file_id > 0) {
                        std::vector<query_curve> curves_cur = load_query_curves(
                            "curve_debug/curve_" + std::to_string(file_id) + ".json");
                        for (const auto& curve : curves_cur) {
                            for (const auto& seg : curve.segments) {
                                if (std::find(id_map_after.begin(), id_map_after.end(), seg.f_id) ==
                                    id_map_after.end()) {
                                    continue;
                                }
                                Eigen::MatrixXd pts(2, 3);
                                for (int i = 0; i < 2; i++) {
                                    Eigen::Vector3d p(0, 0, 0);
                                    for (int j = 0; j < 3; j++) {
                                        p += V_cur.row(seg.fv_ids[j]) * seg.bcs[i](j);
                                    }
                                    pts.row(i) = p;
                                }
                                viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                                viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                                viewer.data().add_edges(
                                    pts.row(0),
                                    pts.row(1),
                                    Eigen::RowVector3d(1, 0, 0));
                            }
                        }
                    }
                }
            }
        } else {
            viewer.data().clear();
            viewer.data().set_mesh(V_in, F_in);
            // viewer.data().point_size /= 3;
            for (const auto curve_origin : curves_in) {
                for (const auto& seg : curve_origin.segments) {
                    Eigen::MatrixXd pts(2, 3);
                    for (int i = 0; i < 2; i++) {
                        Eigen::Vector3d p(0, 0, 0);
                        for (int j = 0; j < 3; j++) {
                            p += V_in.row(seg.fv_ids[j]) * seg.bcs[i](j);
                        }
                        pts.row(i) = p;
                    }
                    viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
                    viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
                    viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
                }
            }
        }

        return false;
    };

    igl::opengl::glfw::Viewer viewer;

    // read obj from file
    viewer.data().set_mesh(V_in, F_in);
    viewer.data().point_size /= 3;
    for (const auto curve_origin : curves_in) {
        for (const auto& seg : curve_origin.segments) {
            Eigen::MatrixXd pts(2, 3);
            for (int i = 0; i < 2; i++) {
                Eigen::Vector3d p(0, 0, 0);
                for (int j = 0; j < 3; j++) {
                    p += V_in.row(seg.fv_ids[j]) * seg.bcs[i](j);
                }
                pts.row(i) = p;
            }
            viewer.data().add_points(pts.row(0), Eigen::RowVector3d(1, 0, 0));
            viewer.data().add_points(pts.row(1), Eigen::RowVector3d(1, 0, 0));
            viewer.data().add_edges(pts.row(0), pts.row(1), Eigen::RowVector3d(1, 0, 0));
        }
    }
    viewer.callback_key_down = key_down;
    viewer.launch();
}
#endif
#include <fstream>
#include <iostream>
#include <sstream>

#include "track_operations.hpp"
// Function to save a vector<query_curve> to a file
void save_query_curves(const std::vector<query_curve>& curves, const std::string& filename)
{
    std::ofstream ofs(filename, std::ios::binary);

    // Write number of curves
    size_t num_curves = curves.size();
    ofs.write(reinterpret_cast<const char*>(&num_curves), sizeof(num_curves));

    // Save each query_curve
    for (const auto& curve : curves) {
        // Write number of segments
        size_t num_segments = curve.segments.size();
        ofs.write(reinterpret_cast<const char*>(&num_segments), sizeof(num_segments));

        // Write each segment
        for (const auto& segment : curve.segments) {
            ofs.write(reinterpret_cast<const char*>(&segment.f_id), sizeof(segment.f_id));
            ofs.write(reinterpret_cast<const char*>(segment.bcs[0].data()), sizeof(segment.bcs[0]));
            ofs.write(reinterpret_cast<const char*>(segment.bcs[1].data()), sizeof(segment.bcs[1]));
            ofs.write(reinterpret_cast<const char*>(segment.fv_ids.data()), sizeof(segment.fv_ids));
        }

        // Write next_segment_ids
        size_t num_next_ids = curve.next_segment_ids.size();
        ofs.write(reinterpret_cast<const char*>(&num_next_ids), sizeof(num_next_ids));
        ofs.write(
            reinterpret_cast<const char*>(curve.next_segment_ids.data()),
            num_next_ids * sizeof(int));
    }

    ofs.close();
}

// Function to load a vector<query_curve> from a file
std::vector<query_curve> load_query_curves(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    std::vector<query_curve> curves;

    // Read number of curves
    size_t num_curves;
    ifs.read(reinterpret_cast<char*>(&num_curves), sizeof(num_curves));
    curves.resize(num_curves);

    // Load each query_curve
    for (auto& curve : curves) {
        // Read number of segments
        size_t num_segments;
        ifs.read(reinterpret_cast<char*>(&num_segments), sizeof(num_segments));
        curve.segments.resize(num_segments);

        // Read each segment
        for (auto& segment : curve.segments) {
            ifs.read(reinterpret_cast<char*>(&segment.f_id), sizeof(segment.f_id));
            ifs.read(reinterpret_cast<char*>(segment.bcs[0].data()), sizeof(segment.bcs[0]));
            ifs.read(reinterpret_cast<char*>(segment.bcs[1].data()), sizeof(segment.bcs[1]));
            ifs.read(reinterpret_cast<char*>(segment.fv_ids.data()), sizeof(segment.fv_ids));
        }

        // Read next_segment_ids
        size_t num_next_ids;
        ifs.read(reinterpret_cast<char*>(&num_next_ids), sizeof(num_next_ids));
        curve.next_segment_ids.resize(num_next_ids);
        ifs.read(
            reinterpret_cast<char*>(curve.next_segment_ids.data()),
            num_next_ids * sizeof(int));
    }

    ifs.close();
    return curves;
}

// TODO: Rational Version of this code
Eigen::Vector3d ComputeBarycentricCoordinates3D(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c,
    double eps = 1e-3)
{
    Eigen::Vector3d v0 = b - a, v1 = c - a, v2 = p - a;
    Eigen::Vector3d n = v0.cross(v1);
    n.normalized();
    if (std::abs(n.dot(p - a)) > eps) {
        // std::cout << "not on the plane, " << std::abs(n.dot(p - a)) << std::endl;
        // not on this face
        return Eigen::Vector3d(-1, -1, -1);
    }
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    // TODO : check why we need this trick
    if (abs(u) < 1e-12) u = 0;
    if (abs(v) < 1e-12) v = 0;
    if (abs(w) < 1e-12) w = 0;
    return Eigen::Vector3d(u, v, w) / (u + v + w);
}

Eigen::Vector3d ComputeBarycentricCoordinates2D(
    const Eigen::Vector2d& p,
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    Eigen::Vector2d v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    return Eigen::Vector3d(u, v, w);
}

// Rational version of ComputeBarycentricCoordinates2D
Eigen::Vector3<wmtk::Rational> ComputeBarycentricCoordinates2D_r(
    const Eigen::Vector2<wmtk::Rational>& p,
    const Eigen::Vector2<wmtk::Rational>& a,
    const Eigen::Vector2<wmtk::Rational>& b,
    const Eigen::Vector2<wmtk::Rational>& c)
{
    Eigen::Vector2<wmtk::Rational> v0 = b - a, v1 = c - a, v2 = p - a;
    wmtk::Rational d00 = v0.dot(v0);
    wmtk::Rational d01 = v0.dot(v1);
    wmtk::Rational d11 = v1.dot(v1);
    wmtk::Rational d20 = v2.dot(v0);
    wmtk::Rational d21 = v2.dot(v1);
    wmtk::Rational denom = d00 * d11 - d01 * d01;
    wmtk::Rational v = (d11 * d20 - d01 * d21) / denom;
    wmtk::Rational w = (d00 * d21 - d01 * d20) / denom;
    wmtk::Rational u = wmtk::Rational(1.0) - v - w;


    // TODO: check why we need this trick
    if (abs(u) < 1e-12) u = wmtk::Rational(0);
    if (abs(v) < 1e-12) v = wmtk::Rational(0);
    if (abs(w) < 1e-12) w = wmtk::Rational(0);

    return Eigen::Vector3<wmtk::Rational>(u, v, w) / (u + v + w);
}


void handle_collapse_edge(
    const Eigen::MatrixXd& UV_joint,
    const Eigen::MatrixXi& F_before,
    const Eigen::MatrixXi& F_after,
    const std::vector<int64_t>& v_id_map_joint,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& id_map_after,
    std::vector<query_point>& query_points,
    bool use_rational)
{
    if (use_rational) {
        handle_collapse_edge_r(
            UV_joint,
            F_before,
            F_after,
            v_id_map_joint,
            id_map_before,
            id_map_after,
            query_points);
        return;
    }
    // std::cout << "Handling EdgeCollapse" << std::endl;
    for (int id = 0; id < query_points.size(); id++) {
        query_point& qp = query_points[id];
        if (qp.f_id < 0) continue;

        // find if qp is in the id_map_after
        auto it = std::find(id_map_after.begin(), id_map_after.end(), qp.f_id);
        if (it != id_map_after.end()) {
            // std::cout << "find qp: " << qp.f_id << std::endl;
            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl;

            // find the index of qp in id_map_after
            int local_index_in_f_after = std::distance(id_map_after.begin(), it);
            // offset of the qp.fv_ids
            int offset_in_f_after = -1;
            for (int i = 0; i < 3; i++) {
                if (v_id_map_joint[F_after(local_index_in_f_after, i)] == qp.fv_ids[0]) {
                    offset_in_f_after = i;
                    break;
                }
            }
            if (offset_in_f_after == -1) {
                // print the whole v_id_map_joint
                std::cout << "something is wrong!" << std::endl;

                std::cout << "the F_after:\n" << F_after << std::endl;

                std::cout << "v_id_map_joint: ";
                for (int i = 0; i < v_id_map_joint.size(); i++) {
                    std::cout << v_id_map_joint[i] << ", ";
                }

                std::cout << "local_index_in_f_after: " << local_index_in_f_after << std::endl;
                for (int i = 0; i < 3; i++) {
                    std::cout << F_after(local_index_in_f_after, i) << ", ";
                }
                for (int i = 0; i < 3; i++) {
                    std::cout << v_id_map_joint[F_after(local_index_in_f_after, i)] << ", ";
                }
                std::cout << std::endl;
                for (int i = 0; i < 3; i++) {
                    std::cout << qp.fv_ids[i] << ", ";
                }
                std::cout << std::endl;
                throw std::runtime_error("something is wrong with offset_in_f_after!");
                continue;
                // return;
            }

            // compute the location of the qp
            Eigen::Vector2d p(0, 0);
            for (int i = 0; i < 3; i++) {
                p += UV_joint.row(F_after(local_index_in_f_after, (i + offset_in_f_after) % 3)) *
                     qp.bc(i);
            }

            // std::cout << "p:\n" << p << std::endl;

            // compute bc of the p in (V, F)_before
            int local_index_in_f_before = -1;
            double bc_min_coef = 1;
            bool bc_updated = false;
            for (int i = 0; i < F_before.rows(); i++) {
                Eigen::Vector3d bc;

                bc = ComputeBarycentricCoordinates2D(
                    p,
                    UV_joint.row(F_before(i, 0)),
                    UV_joint.row(F_before(i, 1)),
                    UV_joint.row(F_before(i, 2)));

                // std::cout << "bc candidate:" << bc << std::endl;
                // std::cout << "check with p:\n"
                //           << bc(0) * UV_joint.row(F_before(i, 0)) +
                //                  bc(1) * UV_joint.row(F_before(i, 1)) +
                //                  bc(2) * UV_joint.row(F_before(i, 2))
                //           << std::endl;
                if (-bc.minCoeff() < bc_min_coef) {
                    // std::cout << "good!" << std::endl;
                    bc_min_coef = -bc.minCoeff();
                    local_index_in_f_before = i;
                    qp.bc = bc;
                    bc_updated = true;
                }
            }

            if (!bc_updated) {
                std::cout << "bc not updated\n" << std::endl;
                continue;
                // return;
            }
            // else {
            //     std::cout << "bc updated\n" << std::endl;
            // }

            // update qp
            qp.f_id = id_map_before[local_index_in_f_before];
            for (int i = 0; i < 3; i++) {
                qp.fv_ids[i] = v_id_map_joint[F_before(local_index_in_f_before, i)];
            }
            // avoid numerical issue
            qp.bc[0] = std::max(0.0, std::min(1.0, qp.bc[0]));
            qp.bc[1] = std::max(0.0, std::min(1.0, qp.bc[1]));
            qp.bc[2] = std::max(0.0, std::min(1.0, qp.bc[2]));
            qp.bc /= qp.bc.sum();

            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl;
        }
    }
}

// Rational version of handle_collapse_edge
void handle_collapse_edge_r(
    const Eigen::MatrixXd& UV_joint,
    const Eigen::MatrixXi& F_before,
    const Eigen::MatrixXi& F_after,
    const std::vector<int64_t>& v_id_map_joint,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& id_map_after,
    std::vector<query_point>& query_points)
{
    // std::cout << "Handling EdgeCollapse Rational" << std::endl;
    // first convert the UV_joint to Rational
    Eigen::MatrixX<wmtk::Rational> UV_joint_r(UV_joint.rows(), UV_joint.cols());
    for (int i = 0; i < UV_joint.rows(); i++) {
        for (int j = 0; j < UV_joint.cols(); j++) {
            UV_joint_r(i, j) = wmtk::Rational(UV_joint(i, j));
        }
    }


    for (int id = 0; id < query_points.size(); id++) {
        query_point& qp = query_points[id];
        if (qp.f_id < 0) continue;


        // find if qp is in the id_map_after
        auto it = std::find(id_map_after.begin(), id_map_after.end(), qp.f_id);
        if (it != id_map_after.end()) {
            // std::cout << "find qp: " << qp.f_id << std::endl;
            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl;
            // find the index of qp in id_map_after
            int local_index_in_f_after = std::distance(id_map_after.begin(), it);
            // offset of the qp.fv_ids
            int offset_in_f_after = -1;
            for (int i = 0; i < 3; i++) {
                if (v_id_map_joint[F_after(local_index_in_f_after, i)] == qp.fv_ids[0]) {
                    offset_in_f_after = i;
                    break;
                }
            }
            if (offset_in_f_after == -1) {
                std::cout << "qp.fv_id: " << qp.fv_ids[0] << ", " << qp.fv_ids[1] << ", "
                          << qp.fv_ids[2] << std::endl;
                std::cout << "local_index_in_f_after: " << local_index_in_f_after << std::endl;
                std::cout << "something is wrong!" << std::endl;
                continue;
                // return;
            }

            // compute the location of the qp
            Eigen::Vector2<wmtk::Rational> p(0, 0);
            for (int i = 0; i < 3; i++) {
                p = p + UV_joint_r.row(F_after(local_index_in_f_after, (i + offset_in_f_after) % 3))
                                .transpose() *
                            wmtk::Rational(qp.bc(i));
            }


            // compute bc of the p in (V, F)_before
            int local_index_in_f_before = -1;

            // if computation is exact we don't need this anymore
            // double bc_min_coef = 1;
            bool bc_updated = false;
            for (int i = 0; i < F_before.rows(); i++) {
                Eigen::Vector3<wmtk::Rational> bc_rational;

                bc_rational = ComputeBarycentricCoordinates2D_r(
                    p,
                    UV_joint_r.row(F_before(i, 0)),
                    UV_joint_r.row(F_before(i, 1)),
                    UV_joint_r.row(F_before(i, 2)));

                // std::cout << "bc candidate:" << bc_rational(0).to_double() << ", "
                //           << bc_rational(1).to_double() << ", " << bc_rational(2).to_double()
                //           << std::endl;

                if (bc_rational.minCoeff() >= 0 && bc_rational.maxCoeff() <= 1) {
                    local_index_in_f_before = i;
                    qp.bc(0) = bc_rational(0).to_double();
                    qp.bc(1) = bc_rational(1).to_double();
                    qp.bc(2) = bc_rational(2).to_double();
                    bc_updated = true;
                }
            }

            if (!bc_updated) {
                std::cout << "bc not updated in rational\n" << std::endl;
                throw std::runtime_error("bc not updated in rational");
                continue;
                // return;
            }

            // update qp
            qp.f_id = id_map_before[local_index_in_f_before];
            for (int i = 0; i < 3; i++) {
                qp.fv_ids[i] = v_id_map_joint[F_before(local_index_in_f_before, i)];
            }
        }
    }
}

bool intersectSegmentEdge(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c,
    const Eigen::Vector2d& d,
    Eigen::Vector2d& barycentric,
    double eps = 1e-8,
    bool debug_mode = false)
{
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d cd = d - c;
    Eigen::Vector2d ac = c - a;

    double denominator = (ab.x() * cd.y() - ab.y() * cd.x());
    if (std::abs(denominator / ab.norm() / cd.norm()) < 1e-8) return false; // parallel

    double t = (ac.x() * cd.y() - ac.y() * cd.x()) / denominator;
    double u = -(ab.x() * ac.y() - ab.y() * ac.x()) / denominator;
    double alpha = 1 - u;
    double beta = u;
    barycentric = Eigen::Vector2d(alpha, beta);
    if (debug_mode) std::cout << "t: " << t << ", u: " << u << std::endl;
    if (t >= 0 && t <= 1 + eps && u >= 0 && u <= 1 + eps) {
        if (debug_mode) std::cout << "intersection found" << std::endl;
        alpha = std::max(0.0, std::min(1.0, alpha));
        beta = std::max(0.0, std::min(1.0, beta));
        barycentric = Eigen::Vector2d(alpha, beta) / (alpha + beta);
        return true;
    }
    if (debug_mode) std::cout << "no intersection" << std::endl;
    // std::cout << "t: " << t << ", u: " << u << std::endl;
    return false;
}

bool intersectSegmentEdge_r(
    const Eigen::Vector2<wmtk::Rational>& a,
    const Eigen::Vector2<wmtk::Rational>& b,
    const Eigen::Vector2<wmtk::Rational>& c,
    const Eigen::Vector2<wmtk::Rational>& d,
    Eigen::Vector2<wmtk::Rational>& barycentric,
    bool debug_mode = false)
{
    Eigen::Vector2<wmtk::Rational> ab = b - a;
    Eigen::Vector2<wmtk::Rational> cd = d - c;
    Eigen::Vector2<wmtk::Rational> ac = c - a;

    wmtk::Rational denominator = (ab.x() * cd.y() - ab.y() * cd.x());
    if (denominator == wmtk::Rational(0)) {
        if (debug_mode) std::cout << "parallel" << std::endl;
        return false; // parallel
    }
    wmtk::Rational t = (ac.x() * cd.y() - ac.y() * cd.x()) / denominator;
    wmtk::Rational u = -(ab.x() * ac.y() - ab.y() * ac.x()) / denominator;
    if (debug_mode) std::cout << "t: " << t << ", u: " << u << std::endl;

    // TODO: check why this is needed
    double eps = 1e-10;
    if (t >= 0 && t <= wmtk::Rational(1 + eps) && u >= 0 && u <= 1) {
        // Calculate barycentric coordinates for intersection
        wmtk::Rational alpha = 1 - u;
        wmtk::Rational beta = u;
        barycentric = Eigen::Vector2<wmtk::Rational>(alpha, beta);
        if (debug_mode) std::cout << "intersection found" << std::endl;
        return true;
    }
    if (debug_mode) std::cout << "no intersection" << std::endl;
    return false;
}

void handle_one_segment(
    query_curve& curve,
    int id,
    std::vector<query_point>& query_points,
    const Eigen::MatrixXd& UV_joint,
    const Eigen::MatrixXi& F_before,
    const std::vector<int64_t>& v_id_map_joint,
    const std::vector<int64_t>& id_map_before,
    Eigen::MatrixXi& TT,
    Eigen::MatrixXi& TTi)
{
    query_segment& qs = curve.segments[id];

    if (query_points[0].f_id == query_points[1].f_id) {
        // two endpoints are on the same face --> no need to compute intersections
        qs.f_id = query_points[0].f_id;
        qs.bcs[0] = query_points[0].bc;
        qs.bcs[1] = query_points[1].bc;
        qs.fv_ids = query_points[0].fv_ids;
    } else {
        if (TT.rows() == 0) {
            igl::triangle_triangle_adjacency(F_before, TT, TTi);
        }
        // TODO: maybe not need eps any more
        double eps = 1e-8;

        // convert the UV_Joint to Rational
        Eigen::MatrixX<wmtk::Rational> UV_joint_r(UV_joint.rows(), UV_joint.cols());
        for (int i = 0; i < UV_joint.rows(); i++) {
            for (int j = 0; j < UV_joint.cols(); j++) {
                UV_joint_r(i, j) = wmtk::Rational(UV_joint(i, j));
            }
        }

        // cache for the last new segment
        int old_next_seg = curve.next_segment_ids[id];

        // compute the two end points
        auto it = std::find(id_map_before.begin(), id_map_before.end(), query_points[0].f_id);
        if (it == id_map_before.end()) {
            std::cout << "query_points[0] not found" << std::endl;
            throw std::runtime_error("query_points[0] not found");
        }
        int current_fid = std::distance(id_map_before.begin(), it);
        Eigen::Vector2<wmtk::Rational> p0(0, 0);
        for (int i = 0; i < 3; i++) {
            p0 = p0 + UV_joint_r.row(F_before(current_fid, i)).transpose() *
                          wmtk::Rational(query_points[0].bc(i));
        }
        it = std::find(id_map_before.begin(), id_map_before.end(), query_points[1].f_id);
        if (it == id_map_before.end()) {
            std::cout << "query_points[1] not found" << std::endl;
            throw std::runtime_error("query_points[1] not found");
        }
        int target_fid = std::distance(id_map_before.begin(), it);
        Eigen::Vector2<wmtk::Rational> p1(0, 0);
        for (int i = 0; i < 3; i++) {
            p1 = p1 + UV_joint_r.row(F_before(target_fid, i)).transpose() *
                          wmtk::Rational(query_points[1].bc(i));
        }

        int current_edge_id = -1;
        // case that the first point is on the edge
        for (int eid = 0; eid < 3; eid++) {
            if (query_points[0].bc(eid) == 0) {
                current_edge_id = (eid + 1) % 3;
            }
        }

        Eigen::Vector2<wmtk::Rational> last_edge_bc;
        {
            int next_edge_id0 = -1;
            // find the first intersection
            for (int edge_id = 0; edge_id < 3; edge_id++) {
                if (edge_id == current_edge_id) continue; // we need to find the other intersection

                Eigen::VectorX<wmtk::Rational> a = UV_joint_r.row(F_before(current_fid, edge_id));
                Eigen::VectorX<wmtk::Rational> b =
                    UV_joint_r.row(F_before(current_fid, (edge_id + 1) % 3));

                Eigen::Vector2<wmtk::Rational> edge_bc;
                if (intersectSegmentEdge_r(p0, p1, a, b, edge_bc)) {
                    next_edge_id0 = edge_id;
                    qs.f_id = query_points[0].f_id;
                    qs.bcs[0] = query_points[0].bc;
                    qs.bcs[1] << 0, 0, 0;
                    qs.bcs[1](edge_id) = edge_bc(0).to_double();
                    qs.bcs[1]((edge_id + 1) % 3) = edge_bc(1).to_double();
                    qs.fv_ids = query_points[0].fv_ids;
                    last_edge_bc = edge_bc;
                    curve.next_segment_ids[id] =
                        curve.segments.size(); // next segment will add to the end

                    break;
                }
            }

            if (next_edge_id0 == -1) {
                // case that the first point is not on a edge, then it is a bug
                if (current_edge_id == -1) {
                    std::cout << "no first intersection found" << std::endl;
                    throw std::runtime_error("no first intersection found");
                }
                // solve this by changing the first point
                int e0 = current_edge_id;
                int f1 = TT(current_fid, e0);
                int e1 = TTi(current_fid, e0);

                query_points[0].f_id = id_map_before[f1];
                query_points[0].fv_ids << v_id_map_joint[F_before(f1, 0)],
                    v_id_map_joint[F_before(f1, 1)], v_id_map_joint[F_before(f1, 2)];
                auto current_bc = query_points[0].bc;
                query_points[0].bc(e1) = current_bc((e0 + 1) % 3);
                query_points[0].bc((e1 + 1) % 3) = current_bc(e0);
                query_points[0].bc((e1 + 2) % 3) = 0;

                std::cout << "try change start p0's fid and solve" << std::endl;
                handle_one_segment(
                    curve,
                    id,
                    query_points,
                    UV_joint,
                    F_before,
                    v_id_map_joint,
                    id_map_before,
                    TT,
                    TTi);
                return;

                std::cout << "no intersection found place 1" << std::endl;
                throw std::runtime_error("no intersection found");
            }

            current_edge_id = TTi(current_fid, next_edge_id0);
            current_fid = TT(current_fid, next_edge_id0);
        }

        // find the rest intersections
        while (current_fid != target_fid) {
            // std::cout << "current_fid: " << current_fid << std::endl;
            // std::cout << "target_fid: " << target_fid << std::endl;

            int next_edge_id = -1;
            for (int edge_id = 0; edge_id < 3; edge_id++) {
                if (edge_id == current_edge_id) continue;
                Eigen::VectorX<wmtk::Rational> a = UV_joint_r.row(F_before(current_fid, edge_id));
                Eigen::VectorX<wmtk::Rational> b =
                    UV_joint_r.row(F_before(current_fid, (edge_id + 1) % 3));
                Eigen::Vector2<wmtk::Rational> edge_bc;
                query_segment qs_new;

                if (intersectSegmentEdge_r(p0, p1, a, b, edge_bc)) {
                    next_edge_id = edge_id;
                    qs_new.f_id = id_map_before[current_fid];
                    qs_new.bcs[0] = Eigen::Vector3d(0, 0, 0);
                    qs_new.bcs[1] = Eigen::Vector3d(0, 0, 0);

                    qs_new.bcs[0](current_edge_id) = last_edge_bc(1).to_double();
                    qs_new.bcs[0]((current_edge_id + 1) % 3) = last_edge_bc(0).to_double();
                    qs_new.bcs[1](next_edge_id) = edge_bc(0).to_double();
                    qs_new.bcs[1]((next_edge_id + 1) % 3) = edge_bc(1).to_double();

                    qs_new.fv_ids << v_id_map_joint[F_before(current_fid, 0)],
                        v_id_map_joint[F_before(current_fid, 1)],
                        v_id_map_joint[F_before(current_fid, 2)];

                    // push the new segment
                    curve.segments.push_back(qs_new);
                    curve.next_segment_ids.push_back(curve.next_segment_ids.size() + 1);


                    // record the last edge
                    last_edge_bc = edge_bc;
                    break;
                }
            }

            if (next_edge_id == -1) {
                std::cout << "no intersection found place 2" << std::endl;
                throw std::runtime_error("no intersection found");
                continue;
            }

            current_edge_id = TTi(current_fid, next_edge_id);
            current_fid = TT(current_fid, next_edge_id);
        }

        // add last segment
        {
            query_segment qs_new;
            qs_new.f_id = query_points[1].f_id;
            qs_new.bcs[0] = Eigen::Vector3d(0, 0, 0);
            qs_new.bcs[1] = query_points[1].bc;
            qs_new.bcs[0](current_edge_id) = last_edge_bc(1).to_double();
            qs_new.bcs[0]((current_edge_id + 1) % 3) = last_edge_bc(0).to_double();
            qs_new.fv_ids = query_points[1].fv_ids;

            // push the new segment
            curve.segments.push_back(qs_new);
            curve.next_segment_ids.push_back(old_next_seg);
        }

        // std::cout << "finish one segment\n" << std::endl;
    }
}


void handle_collapse_edge_curve(
    const Eigen::MatrixXd& UV_joint,
    const Eigen::MatrixXi& F_before,
    const Eigen::MatrixXi& F_after,
    const std::vector<int64_t>& v_id_map_joint,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& id_map_after,
    query_curve& curve,
    bool use_rational)
{
    std::cout << "Handling EdgeCollapse curve" << std::endl;
    int curve_length = curve.segments.size();
    Eigen::MatrixXi TT, TTi;

    for (int id = 0; id < curve_length; id++) {
        ////////////////////////////////////
        // map the two end points of one segment
        ////////////////////////////////////
        query_segment& qs = curve.segments[id];
        query_point qp0 = {qs.f_id, qs.bcs[0], qs.fv_ids};
        query_point qp1 = {qs.f_id, qs.bcs[1], qs.fv_ids};
        std::vector<query_point> query_points = {qp0, qp1};
        auto query_points_copy = query_points;

        handle_collapse_edge(
            UV_joint,
            F_before,
            F_after,
            v_id_map_joint,
            id_map_before,
            id_map_after,
            query_points);
        if (query_points[0].bc.minCoeff() < 1e-8 || query_points[0].bc.minCoeff() < 1e-8) {
            query_points = query_points_copy;
            handle_collapse_edge_r(
                UV_joint,
                F_before,
                F_after,
                v_id_map_joint,
                id_map_before,
                id_map_after,
                query_points);
        }
        handle_one_segment(
            curve,
            id,
            query_points,
            UV_joint,
            F_before,
            v_id_map_joint,
            id_map_before,
            TT,
            TTi);
    }
}


void handle_split_edge(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& F_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& F_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    std::vector<query_point>& query_points)
{
    std::cout << "Handling EdgeSplit" << std::endl;
    // igl::parallel_for(query_points.size(), [&](int id) {
    for (int id = 0; id < query_points.size(); id++) {
        query_point& qp = query_points[id];
        if (qp.f_id < 0) continue;
        // find if qp is in the id_map_after
        auto it = std::find(id_map_after.begin(), id_map_after.end(), qp.f_id);
        if (it != id_map_after.end()) {
            // std::cout << "find qp: " << qp.f_id << std::endl;
            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl;

            // find the index of qp in id_map_after
            int local_index_in_f_after = std::distance(id_map_after.begin(), it);
            // offset of the qp.fv_ids
            int offset_in_f_after = -1;
            for (int i = 0; i < 3; i++) {
                if (v_id_map_after[F_after(local_index_in_f_after, i)] == qp.fv_ids[0]) {
                    offset_in_f_after = i;
                    break;
                }
            }
            if (offset_in_f_after == -1) {
                std::cout << "something is wrong!" << std::endl;
                continue;
                // return;
            }

            // compute the location of the qp
            int V_cols = V_after.cols();
            Eigen::VectorXd p = Eigen::VectorXd::Zero(V_cols);
            for (int i = 0; i < 3; i++) {
                p += V_after.row(F_after(local_index_in_f_after, (i + offset_in_f_after) % 3)) *
                     qp.bc(i);
            }


            // compute bc of the p in (V, F)_before
            int local_index_in_f_before = -1;
            double bc_min_coef = 1;
            bool bc_updated = false;
            for (int i = 0; i < F_before.rows(); i++) {
                Eigen::Vector3d bc;
                if (V_cols == 2) {
                    bc = ComputeBarycentricCoordinates2D(
                        p,
                        V_before.row(F_before(i, 0)),
                        V_before.row(F_before(i, 1)),
                        V_before.row(F_before(i, 2)));
                } else // V_cols == 3
                {
                    bc = ComputeBarycentricCoordinates3D(
                        p,
                        V_before.row(F_before(i, 0)),
                        V_before.row(F_before(i, 1)),
                        V_before.row(F_before(i, 2)));
                }
                if (-bc.minCoeff() < bc_min_coef) {
                    bc_min_coef = -bc.minCoeff();
                    local_index_in_f_before = i;
                    qp.bc = bc;
                    bc_updated = true;
                }
            }

            if (!bc_updated) {
                std::cout << "bc not updated\n" << std::endl;
                continue;
                // return;
            }

            // update qp
            qp.f_id = id_map_before[local_index_in_f_before];
            for (int i = 0; i < 3; i++) {
                qp.fv_ids[i] = v_id_map_before[F_before(local_index_in_f_before, i)];
            }
            qp.bc[0] = std::max(0.0, std::min(1.0, qp.bc[0]));
            qp.bc[1] = std::max(0.0, std::min(1.0, qp.bc[1]));
            qp.bc[2] = std::max(0.0, std::min(1.0, qp.bc[2]));
            qp.bc /= qp.bc.sum();

            // std::cout << "qp-> " << qp.f_id << std::endl;
            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl
            //           << std::endl;
        }
    }
    // });
}

void handle_swap_edge(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& F_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& F_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    std::vector<query_point>& query_points)
{
    // std::cout << "Handling EdgeSwap" << std::endl;
    // igl::parallel_for(query_points.size(), [&](int id) {
    for (int id = 0; id < query_points.size(); id++) {
        query_point& qp = query_points[id];
        if (qp.f_id < 0) continue;
        // find if qp is in the id_map_after
        auto it = std::find(id_map_after.begin(), id_map_after.end(), qp.f_id);
        if (it != id_map_after.end()) {
            // std::cout << "find qp: " << qp.f_id << std::endl;
            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl;

            // find the index of qp in id_map_after
            int local_index_in_f_after = std::distance(id_map_after.begin(), it);
            // offset of the qp.fv_ids
            int offset_in_f_after = -1;
            for (int i = 0; i < 3; i++) {
                if (v_id_map_after[F_after(local_index_in_f_after, i)] == qp.fv_ids[0]) {
                    offset_in_f_after = i;
                    break;
                }
            }
            if (offset_in_f_after == -1) {
                std::cout << "something is wrong!" << std::endl;
                continue;
                // return;
            }

            // compute the location of the qp
            int V_cols = V_after.cols();
            Eigen::VectorXd p = Eigen::VectorXd::Zero(V_cols);
            for (int i = 0; i < 3; i++) {
                p += V_after.row(F_after(local_index_in_f_after, (i + offset_in_f_after) % 3)) *
                     qp.bc(i);
            }


            // compute bc of the p in (V, F)_before
            int local_index_in_f_before = -1;
            double bc_min_coef = 1;
            bool bc_updated = false;
            for (int i = 0; i < F_before.rows(); i++) {
                Eigen::Vector3d bc;
                if (V_cols == 2) {
                    bc = ComputeBarycentricCoordinates2D(
                        p,
                        V_before.row(F_before(i, 0)),
                        V_before.row(F_before(i, 1)),
                        V_before.row(F_before(i, 2)));
                } else // V_cols == 3
                {
                    bc = ComputeBarycentricCoordinates3D(
                        p,
                        V_before.row(F_before(i, 0)),
                        V_before.row(F_before(i, 1)),
                        V_before.row(F_before(i, 2)),
                        100000);
                }
                if (-bc.minCoeff() < bc_min_coef) {
                    bc_min_coef = -bc.minCoeff();
                    local_index_in_f_before = i;
                    qp.bc = bc;
                    bc_updated = true;
                }
            }

            if (!bc_updated) {
                std::cout << "bc not updated\n" << std::endl;
                continue;
                // return;
            }

            // update qp
            qp.f_id = id_map_before[local_index_in_f_before];
            for (int i = 0; i < 3; i++) {
                qp.fv_ids[i] = v_id_map_before[F_before(local_index_in_f_before, i)];
            }
            // avoid numerical issue
            qp.bc[0] = std::max(0.0, std::min(1.0, qp.bc[0]));
            qp.bc[1] = std::max(0.0, std::min(1.0, qp.bc[1]));
            qp.bc[2] = std::max(0.0, std::min(1.0, qp.bc[2]));
            qp.bc /= qp.bc.sum();

            // std::cout << "qp-> " << qp.f_id << std::endl;
            // std::cout << "qp.bc: (" << qp.bc(0) << ", " << qp.bc(1) << ", " << qp.bc(2) << ")"
            //           << std::endl
            //           << std::endl;
        }
    }
    // });
}

void handle_swap_edge_curve(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& F_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& F_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    query_curve& curve)
{
    double eps = 1e-8;
    std::cout << "Handling swap/smooth curve" << std::endl;
    int curve_length = curve.segments.size();
    Eigen::MatrixXi TT, TTi;

    for (int id = 0; id < curve_length; id++) {
        query_segment& qs = curve.segments[id];
        query_point qp0 = {qs.f_id, qs.bcs[0], qs.fv_ids};
        query_point qp1 = {qs.f_id, qs.bcs[1], qs.fv_ids};
        std::vector<query_point> query_points = {qp0, qp1};

        handle_swap_edge(
            V_before,
            F_before,
            id_map_before,
            v_id_map_before,
            V_after,
            F_after,
            id_map_after,
            v_id_map_after,
            query_points);

        handle_one_segment(
            curve,
            id,
            query_points,
            V_before,
            F_before,
            v_id_map_before,
            id_map_before,
            TT,
            TTi);
    }
}

void parse_consolidate_file(
    const json& operation_log,
    std::vector<int64_t>& face_ids_maps,
    std::vector<int64_t>& vertex_ids_maps)
{
    face_ids_maps = operation_log["new2old"][2].get<std::vector<int64_t>>();
    vertex_ids_maps = operation_log["new2old"][0].get<std::vector<int64_t>>();
}

template <typename Matrix>
Matrix json_to_matrix(const json& js)
{
    int rows = js["rows"];
    int cols = js["values"][0].size();

    Matrix mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = js["values"][i][j];
        }
    }

    return mat;
}

void parse_non_collapse_file(
    const json& operation_log,
    bool& is_skipped,
    Eigen::MatrixXd& V_before,
    Eigen::MatrixXi& F_before,
    std::vector<int64_t>& id_map_before,
    std::vector<int64_t>& v_id_map_before,
    Eigen::MatrixXd& V_after,
    Eigen::MatrixXi& F_after,
    std::vector<int64_t>& id_map_after,
    std::vector<int64_t>& v_id_map_after)
{
    is_skipped = operation_log["is_skipped"].get<bool>();
    if (is_skipped) {
        return;
    }

    F_before = json_to_matrix<Eigen::MatrixXi>(operation_log["F_before"]);
    V_before = json_to_matrix<Eigen::MatrixXd>(operation_log["V_before"]);
    id_map_before = operation_log["F_id_map_before"].get<std::vector<int64_t>>();
    v_id_map_before = operation_log["V_id_map_before"].get<std::vector<int64_t>>();

    F_after = json_to_matrix<Eigen::MatrixXi>(operation_log["F_after"]);
    V_after = json_to_matrix<Eigen::MatrixXd>(operation_log["V_after"]);
    id_map_after = operation_log["F_id_map_after"].get<std::vector<int64_t>>();
    v_id_map_after = operation_log["V_id_map_after"].get<std::vector<int64_t>>();
}

void parse_edge_collapse_file(
    const json& operation_log,
    Eigen::MatrixXd& UV_joint,
    Eigen::MatrixXi& F_before,
    Eigen::MatrixXi& F_after,
    std::vector<int64_t>& v_id_map_joint,
    std::vector<int64_t>& id_map_before,
    std::vector<int64_t>& id_map_after)
{
    UV_joint = json_to_matrix<Eigen::MatrixXd>(operation_log["UV_joint"]);
    F_before = json_to_matrix<Eigen::MatrixXi>(operation_log["F_before"]);
    F_after = json_to_matrix<Eigen::MatrixXi>(operation_log["F_after"]);

    v_id_map_joint = operation_log["v_id_map_joint"].get<std::vector<int64_t>>();
    id_map_before = operation_log["F_id_map_before"].get<std::vector<int64_t>>();
    id_map_after = operation_log["F_id_map_after"].get<std::vector<int64_t>>();
}
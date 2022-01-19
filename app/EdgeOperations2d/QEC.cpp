#include <wmtk/TriMesh.h>
#include <wmtk/utils/VectorUtils.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <wmtk/ExecutionScheduler.hpp>
#include "EdgeOperations2d.h"

using namespace Edge2d;
using namespace wmtk;
// get the quadrix in form of an array of 10 floating point numbers
Eigen::MatrixXd compute_Q_f(const EdgeOperations2d& m, const wmtk::TriMesh::Tuple& f_tuple)
{
    auto conn_indices = m.oriented_tri_vertices(f_tuple);
    Eigen::Vector3d A = m.m_vertex_positions[conn_indices[0].vid()];
    Eigen::Vector3d B = m.m_vertex_positions[conn_indices[1].vid()];
    Eigen::Vector3d C = m.m_vertex_positions[conn_indices[2].vid()];

    Eigen::Vector3d n = ((A - B).cross(C - B)).normalized();
    double a = n(0);
    double b = n(1);
    double c = n(2);
    double d = -n.dot(B);

    std::array<double, 10> Q;
    // a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2,
    Q[0] = a * a;
    Q[1] = a * b;
    Q[2] = a * c;
    Q[3] = a * d;
    Q[4] = b * b;
    Q[5] = b * c;
    Q[6] = b * d;
    Q[7] = c * c;
    Q[8] = c * d;
    Q[9] = d * d;
    return Q;
}


Eigen::MatrixXd compute_Q_v(const EdgeOperations2d& m, const TriMesh::Tuple& v_tuple)
{
    auto conn_tris = m.get_one_ring_tris_for_vertex(v_tuple);
    Eigen::MatrixXd Q{};
    auto Q_t = [](auto& m, auto& f_tuple) {
        auto conn_indices = m.oriented_tri_vertices(f_tuple);
        Eigen::Vector3d A = m.m_vertex_positions[conn_indices[0].vid()];
        Eigen::Vector3d B = m.m_vertex_positions[conn_indices[1].vid()];
        Eigen::Vector3d C = m.m_vertex_positions[conn_indices[2].vid()];

        Eigen::Vector3d n = ((A - B).cross(C - B)).normalized();
        Eigen::Vector4d p;
        p(0) = n(0);
        p(1) = n(1);
        p(2) = n(2);
        p(3) = -n.dot(B);

        return p * p.transpose();
    };
    for (auto tri : conn_tris) {
        Q += Q_t(m, tri);
    }
    return Q;
}

double compute_cost_for_v(const EdgeOperations2d& m, const TriMesh::Tuple& v_tuple)
{
    Eigen::MatrixXd Q = compute_Q_v(m, v_tuple);
    Q += compute_Q_v(m, v_tuple.switch_vertex(m));
    Eigen::Vector4d t(0.0, 0.0, 0.0, 1.0);
    auto v = Q.inverse() * t;

    return v.transpose() * Q * v;
}


bool Edge2d::EdgeOperations2d::collapse_qec()
{
    // find the valid pairs (for each vertex)
    std::vector<TriMesh::Tuple> edges = get_edges();
    auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_edges()) collect_all_ops.emplace_back("edge_collapse", loc);

    auto executor = wmtk::ExecutePass<EdgeOperations2d, wmtk::ExecutionPolicy::kSeq>();
    executor.renew_neighbor_tuples = [](auto& m, auto op, auto& tris) {
        auto edges = m.new_edges_after(tris);
        auto optup = std::vector<std::pair<std::string, TriMesh::Tuple>>();
        for (auto& e : edges) optup.emplace_back(op, e);
        return optup;
    };

    executor.priority = [](auto& m, auto _, auto& e) { return -compute_cost_for_v(m, e); };
    executor.should_process = [](auto& m, auto& ele) {
        auto& [val, op, e] = ele;
        if (val > 0) return false; // priority is negated.
        return true;
    };

    executor(*this, collect_all_ops);
    return true;
}

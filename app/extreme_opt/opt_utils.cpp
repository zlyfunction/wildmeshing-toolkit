#include "ExtremeOpt.h"
#include <igl/local_basis.h>
#include "SYMDIR.h"
#include <igl/cat.h>
#include <igl/grad.h>
#include <wmtk/utils/TupleUtils.hpp>

namespace extremeopt {
void get_grad_op(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& grad_op)
{
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);

    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G, false);

    auto face_proj = [](Eigen::MatrixXd& F) {
        std::vector<Eigen::Triplet<double>> IJV;
        int f_num = F.rows();
        for (int i = 0; i < F.rows(); i++) {
            IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
            IJV.push_back(Eigen::Triplet<double>(i, i + f_num, F(i, 1)));
            IJV.push_back(Eigen::Triplet<double>(i, i + 2 * f_num, F(i, 2)));
        }
        Eigen::SparseMatrix<double> P(f_num, 3 * f_num);
        P.setFromTriplets(IJV.begin(), IJV.end());
        return P;
    };

    Eigen::SparseMatrix<double> Dx = face_proj(F1) * G;
    Eigen::SparseMatrix<double> Dy = face_proj(F2) * G;

    Eigen::SparseMatrix<double> hstack = igl::cat(1, Dx, Dy);
    Eigen::SparseMatrix<double> empty(hstack.rows(), hstack.cols());

    grad_op = igl::cat(1, igl::cat(2, hstack, empty), igl::cat(2, empty, hstack));
}
} // namespace extremeopt

double extremeopt::ExtremeOpt::get_e_max_onering(const Tuple& t)
{
    Eigen::MatrixXd V_local, uv_local, Ji;
    Eigen::MatrixXi F_local;
    Eigen::SparseMatrix<double> G_local;
    get_mesh_onering(t, V_local, uv_local, F_local);
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    return wmtk::symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3)).maxCoeff();
}

double extremeopt::ExtremeOpt::get_e_onering_edge(const Tuple& t)
{
    Eigen::MatrixXd V_local, uv_local, Ji;
    Eigen::MatrixXi F_local;
    Eigen::SparseMatrix<double> G_local;
    get_mesh_onering_edge(t, V_local, uv_local, F_local);
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    Eigen::VectorXd areas;
    igl::doublearea(V_local, F_local, areas);
    return wmtk::compute_energy_from_jacobian(Ji, areas) * areas.sum();
}

void extremeopt::ExtremeOpt::cache_edge_positions(const Tuple& t)
{
    position_cache.local().V1 = vertex_attrs[t.vid(*this)].pos_3d;
    position_cache.local().V2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos_3d;
    position_cache.local().uv1 = vertex_attrs[t.vid(*this)].pos;
    position_cache.local().uv2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos;

    position_cache.local().vid1 = t.vid(*this);
    position_cache.local().vid2 = t.switch_vertex(*this).vid(*this);

    position_cache.local().is_v1_bd = this->is_boundary_vertex(t);
    position_cache.local().is_v2_bd = this->is_boundary_vertex(t.switch_vertex(*this));

    if (position_cache.local().is_v1_bd) {
        auto onering_e = get_one_ring_edges_for_vertex(t);
        int cnt = 0;
        for (auto e : onering_e) {
            if (is_boundary_edge(e)) {
                if (cnt == 0) {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e1 = e;
                    else
                        position_cache.local().bd_e1 = e.switch_vertex(*this);
                } else {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e2 = e;
                    else
                        position_cache.local().bd_e2 = e.switch_vertex(*this);
                }
                cnt++;
            }
            if (cnt == 2) break;
        }
    } else if (position_cache.local().is_v2_bd) {
        auto onering_e = get_one_ring_edges_for_vertex(t.switch_vertex(*this));
        int cnt = 0;
        for (auto e : onering_e) {
            if (is_boundary_edge(e)) {
                if (cnt == 0) {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e1 = e;
                    else
                        position_cache.local().bd_e1 = e.switch_vertex(*this);
                } else {
                    if (e.is_ccw(*this))
                        position_cache.local().bd_e2 = e;
                    else
                        position_cache.local().bd_e2 = e.switch_vertex(*this);
                }
                cnt++;
            }
            if (cnt == 2) break;
        }
    }



    Eigen::MatrixXd V_local, uv_local, Ji;
    Eigen::MatrixXi F_local;
    Eigen::SparseMatrix<double> G_local;
    get_mesh_onering_edge(t, V_local, uv_local, F_local);
    get_grad_op(V_local, F_local, G_local);
    wmtk::jacobian_from_uv(G_local, uv_local, Ji);
    Eigen::VectorXd areas;
    igl::doublearea(V_local, F_local, areas);
    position_cache.local().E_before_collapse = wmtk::compute_energy_from_jacobian(Ji, areas) * areas.sum();    

    std::cout << "before collapse: E = " <<  position_cache.local().E_before_collapse << " area = " << areas.sum() << std::endl;
/*
    double E1, E2;
    E1 = get_e_max_onering(t);
    E2 = get_e_max_onering(t.switch_vertex(*this));
    position_cache.local().E_before_collapse = std::max(E1, E2);
*/
}

std::vector<wmtk::TriMesh::Tuple> extremeopt::ExtremeOpt::new_edges_after(
    const std::vector<wmtk::TriMesh::Tuple>& tris) const
{
    std::vector<wmtk::TriMesh::Tuple> new_edges;

    for (auto t : tris) {
        for (auto j = 0; j < 3; j++) {
            new_edges.push_back(wmtk::TriMesh::tuple_from_edge(t.fid(*this), j));
        }
    }
    wmtk::unique_edge_tuples(*this, new_edges);

    return new_edges;
}
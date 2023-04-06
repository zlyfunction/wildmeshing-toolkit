#include <spdlog/common.h>
#include "ExtremeOpt.h"
#include <fstream>
#include <igl/PI.h>
#include <igl/boundary_loop.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/predicates/predicates.h>
#include "SYMDIR.h"
#include <catch2/catch.hpp>

int check_flip(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& Fn)
{
    int fl = 0;
    for (int i = 0; i < Fn.rows(); i++) {
        Eigen::Matrix<double, 1, 2> a_db(uv(Fn(i, 0), 0), uv(Fn(i, 0), 1));
        Eigen::Matrix<double, 1, 2> b_db(uv(Fn(i, 1), 0), uv(Fn(i, 1), 1));
        Eigen::Matrix<double, 1, 2> c_db(uv(Fn(i, 2), 0), uv(Fn(i, 2), 1));
        if (igl::predicates::orient2d(a_db, b_db, c_db) != igl::predicates::Orientation::POSITIVE) {
            fl++;
        }
    }
    return fl;
}

bool find_edge_in_F(const Eigen::MatrixXi& F, int v0, int v1, int& fid, int& eid)
{
    fid = -1;
    eid = -1;
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            if (F(i, j) == v0 && F(i, (j + 1) % 3) == v1) {
                fid = i;
                eid = 3 - j - ((j + 1) % 3);
                return true;
            }
        }
    }
    return false;
}
void transform_EE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& EE_v,
    std::vector<std::vector<int>>& EE_e)
{
    EE_e.resize(EE_v.rows());
    for (int i = 0; i < EE_v.rows(); i++) {
        std::vector<int> one_row;
        int v0 = EE_v(i, 0), v1 = EE_v(i, 1);
        int fid, eid;
        if (find_edge_in_F(F, v0, v1, fid, eid)) {
            one_row.push_back(fid);
            one_row.push_back(eid);
            // one_row.push_back(3 * fid + eid);
        } else {
            std::cout << "Something Wrong in transform_EE: edge not found in F" << std::endl;
        }

        v0 = EE_v(i, 2);
        v1 = EE_v(i, 3);
        if (find_edge_in_F(F, v0, v1, fid, eid)) {
            one_row.push_back(fid);
            one_row.push_back(eid);
            // one_row.push_back(3 * fid + eid);
        } else {
            std::cout << "Something Wrong in transform_EE: edge not found in F" << std::endl;
        }
        EE_e[i] = one_row;
    }
}

void init_model(extremeopt::ExtremeOpt &extremeopt, const std::string &model, extremeopt::Parameters &param)
{
    param.model_name = model;

    std::string input_file = "../build/objs/" + model + "_init.obj";
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    igl::readOBJ(input_file, V, uv, uv, F, F, F);
    Eigen::MatrixXi EE;
    int EE_rows;
    std::ifstream EE_in("../build/objs/EE/" + model + "_EE.txt");
    EE_in >> EE_rows;
    EE.resize(EE_rows, 4);
    for (int i = 0; i < EE.rows(); i++) {
        EE_in >> EE(i, 0) >> EE(i, 1) >> EE(i, 2) >> EE(i, 3);
    }

    extremeopt.m_params = param;
    extremeopt.create_mesh(V, F, uv);
    std::vector<std::vector<int>> EE_e;
    transform_EE(F, EE, EE_e);
    extremeopt.init_constraints(EE_e);
}

double compute_energy(const Eigen::MatrixXd &V, const Eigen::MatrixXd &uv, const Eigen::MatrixXi &F)
{
    Eigen::SparseMatrix<double> G_global;
    extremeopt::get_grad_op(V, F, G_global);
    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);
    Eigen::MatrixXd Ji;
    wmtk::jacobian_from_uv(G_global, uv, Ji);
    return wmtk::compute_energy_from_jacobian(Ji, dblarea) * dblarea.sum();
}

TEST_CASE("helmet_collapse", "[helmet]")
{
    extremeopt::Parameters param;
    param.max_iters = 1;
    param.local_smooth = false;
    param.global_smooth = false;
    param.use_envelope = true;
    param.elen_alpha = 0.2;
    param.do_projection = false;
    param.with_cons = true;
    param.save_meshes = false;

    std::string model = "helmet";
    extremeopt::ExtremeOpt extremeopt;
    init_model(extremeopt, model, param);
    
    Eigen::MatrixXd V_in, uv_in;
    Eigen::MatrixXi F_in;
    extremeopt.export_mesh(V_in, F_in, uv_in);
    double E_in = compute_energy(V_in, uv_in, F_in);
    
    extremeopt.collapse_all_edges();

        
    Eigen::MatrixXd V_out, uv_out;
    Eigen::MatrixXi F_out;
    extremeopt.export_mesh(V_out, F_out, uv_out);
    double E_out = compute_energy(V_out, uv_out, F_out);
    SECTION( "no flip" ) 
    {
        REQUIRE(check_flip(uv_out, F_out) == 0);
    }
    SECTION( "seamless constraints" ) 
    {
        REQUIRE(extremeopt.check_constraints());   
    }
    SECTION("smaller energy")
    {
        REQUIRE(E_out <= E_in);
    }


}

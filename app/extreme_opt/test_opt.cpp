#include <spdlog/common.h>
#include "ExtremeOpt.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <CLI/CLI.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#include <fstream>
#include <igl/PI.h>
#include <igl/boundary_loop.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>

#include "autodiff_jakob.h"
// using namespace jakob;
DECLARE_DIFFSCALAR_BASE();
using namespace extremeopt;
template <typename T>
T E_from_J(T a, T b, T c, T d)
{
    auto det = a * d - b * c;
    auto frob2 = a * a + b * b + c * c + d * d;
    return frob2 * (1 + 1 / (det * det)); // sym_dir
}


int main(int argc, char** argv)
{
    CLI::App app{argv[0]};
    std::string input_dir = "./objs";
    std::string model = "helmet_v3924_before";
    app.add_option("-i,--input", input_dir, "Input mesh dir.");
    app.add_option("-m,--model", model, "Input model name.");
    CLI11_PARSE(app, argc, argv);
    std::string input_file = input_dir + "/" + model + "_init.obj";
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    igl::readOBJ(input_file, V, uv, uv, F, F, F);
    wmtk::logger()
        .info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());
    Eigen::SparseMatrix<double> G;
    get_grad_op(V, F, G);
    Eigen::VectorXd area;
    igl::doublearea(V, F, area);    
    int vid = 1;

    typedef DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>> DScalar;
    DiffScalarBase::setVariableCount(2);

    Eigen::Matrix<DScalar, -1, -1> G_dense = Eigen::MatrixXd(G).template cast<DScalar>();
    
    Eigen::Matrix<DScalar, -1, -1> uv_x;
    uv_x.resize(uv.rows(), uv.cols());
    for (int i = 0; i < uv.rows(); i++)
    {
        if (i != vid)
        {
            uv_x.row(i) << DScalar(uv(i,0)), DScalar(uv(i,1));
        }
        else
        {
            uv_x.row(i) << DScalar(0, uv(i,0)), DScalar(1, uv(i,1));

        }
    }
    Eigen::Matrix<DScalar, -1, 1> uv_flat = Eigen::Map<Eigen::Matrix<DScalar, -1, 1>>(uv_x.data(), uv_x.size());
    Eigen::Matrix<DScalar, -1, 1> altJ = G_dense * uv_flat;
    Eigen::Matrix<DScalar, -1, -1> Ji = Eigen::Map<Eigen::Matrix<DScalar, -1, -1>>(altJ.data(), altJ.rows() / 4, 4);

    // for (int i = 0; i < Ji.rows(); i++)
    // {
    //     std::cout << "triangle " << i << std::endl;
    //     for (int j = 0; j < Ji.cols(); j++)
    //     {
    //         std::cout << Ji(i, j).getValue() << ",\n" << Ji(i, j).getGradient() << ",\n";
    //     }
    //     std::cout << std::endl;
    // }

    DScalar E(0);
    for (int i = 0; i < Ji.rows(); i++)
    {
        E += area(i) * E_from_J(Ji(i,0),Ji(i,1),Ji(i,2),Ji(i,3));
    }
    E /= area.sum();

    std::cout << "Energy = " << E.getValue() << std::endl;
    std::cout << "Grad =\n " << E.getGradient() << std::endl;
    std::cout << "Hessian = \n" << E.getHessian() << std::endl;

    return 0;
}



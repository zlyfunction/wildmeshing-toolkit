#include "SYMDIR.h"

// From AutoDiff
namespace jakob
{
#include "autodiff_jakob.h"
  DECLARE_DIFFSCALAR_BASE();

  template <typename Scalar>
  Scalar gradient_and_hessian_from_J(const Eigen::Matrix<Scalar, 1, 4> &J,
                                     Eigen::Matrix<Scalar, 1, 4> &local_grad,
                                     Eigen::Matrix<Scalar, 4, 4> &local_hessian)
  {
      typedef DScalar2<Scalar, Eigen::Matrix<Scalar, 4, 1>, Eigen::Matrix<Scalar, 4, 4>> DScalar;
      DiffScalarBase::setVariableCount(4);

      DScalar a(0, J(0));
      DScalar b(1, J(1));
      DScalar c(2, J(2));
      DScalar d(3, J(3));
      auto sd = wmtk::symmetric_dirichlet_energy_t(a, b, c, d);
      local_grad = sd.getGradient();
      local_hessian = sd.getHessian();
      DiffScalarBase::setVariableCount(0);
      return sd.getValue();
  }
} // namespace jakob

namespace wmtk{
    
    template <typename Scalar>
    void jacobian_from_uv(const Eigen::SparseMatrix<Scalar> &G, const Eigen::Matrix<Scalar, -1, -1> &uv, Eigen::Matrix<Scalar, -1, -1> &Ji)
    {
        Eigen::Matrix<Scalar, -1, 1> altJ = G * Eigen::Map<const Eigen::Matrix<Scalar, -1, 1>>(uv.data(), uv.size());
        Ji = Eigen::Map<Eigen::Matrix<Scalar, -1, -1>>(altJ.data(), G.rows() / 4, 4);
    }
    
    template <typename Scalar>
    Scalar compute_energy_from_jacobian(const Eigen::Matrix<Scalar, -1, -1> &J, const Eigen::Matrix<Scalar, -1, 1> &area)
    {
        return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3)).dot(area) / area.sum();
    }
    
    template <typename Scalar>
    Scalar grad_and_hessian_from_jacobian(const Eigen::Matrix<Scalar, -1, 1> &area, const Eigen::Matrix<Scalar, -1, -1> &jacobian,
                                      Eigen::Matrix<Scalar, -1, -1> &total_grad, Eigen::SparseMatrix<Scalar> &hessian, bool with_hessian)
    {
        int f_num = area.rows();
        total_grad.resize(f_num, 4);
        total_grad.setZero();
        Scalar energy = 0;
        hessian.resize(4 * f_num, 4 * f_num);
        std::vector<Eigen::Triplet<Scalar>> IJV;
        IJV.reserve(16 * f_num);
        Scalar total_area = area.sum();

        std::vector<Eigen::Matrix<Scalar, 4, 4>> all_hessian(f_num);

        for (int i = 0; i < f_num; i++)
        {
            Eigen::Matrix<Scalar, 1, 4> J = jacobian.row(i);
            Eigen::Matrix<Scalar, 4, 4> local_hessian;
            Eigen::Matrix<Scalar, 1, 4> local_grad;
            energy += jakob::gradient_and_hessian_from_J(J, local_grad, local_hessian) * area(i) / total_area;

            local_grad *= area(i) / total_area;
            total_grad.row(i) = local_grad;
            if (with_hessian)
            {
                local_hessian *= area(i) / total_area;
                all_hessian[i] = local_hessian;
            }
        }

        if (with_hessian)
        {
            hessian.reserve(Eigen::VectorXi::Constant(4 * f_num, 4));
            for (int i = 0; i < f_num; i++)
            {
                Eigen::Matrix<Scalar, 4, 4> &local_hessian = all_hessian[i];
                // if (fabs(total_grad(i)) > 1e-3)
                    // project_hessian(local_hessian);
                for (int v1 = 0; v1 < 4; v1++)
                {
                    for (int v2 = 0; v2 < v1 + 1; v2++)
                    {
                        hessian.insert(v1 * f_num + i, v2 * f_num + i) = local_hessian(v1, v2);
                        if (v1 != v2)
                            hessian.insert(v2 * f_num + i, v1 * f_num + i) = local_hessian(v1, v2);
                    }
                }
            }
            hessian.makeCompressed();
        }
        return energy;
    }

    template <typename Scalar>
    Scalar get_grad_and_hessian(const Eigen::SparseMatrix<Scalar> &G,
                                const Eigen::Matrix<Scalar, -1, 1> &area,
                                const Eigen::Matrix<Scalar, -1, -1> &uv,
                                Eigen::Matrix<Scalar, -1, 1> &grad,
                                Eigen::SparseMatrix<Scalar> &hessian,
                                bool get_hessian)
    {
        int f_num = area.rows();
        Eigen::Matrix<Scalar, -1, -1> Ji, total_grad;
        jacobian_from_uv(G, uv, Ji);
        Scalar energy;
        energy = grad_and_hessian_from_jacobian(area, Ji, total_grad, hessian, get_hessian);

        Eigen::Matrix<Scalar, -1, 1> vec_grad = Eigen::Map<Eigen::Matrix<Scalar, -1, 1>>(total_grad.data(), total_grad.size());

        hessian = G.transpose() * hessian * G;
        grad = vec_grad.transpose() * G;

        return energy;
    }

    template void jacobian_from_uv<double>(const Eigen::SparseMatrix<double> &, const Eigen::Matrix<double, -1, -1> &, Eigen::Matrix<double, -1, -1> &);
    
    template double get_grad_and_hessian<double>(const Eigen::SparseMatrix<double> &,
                                             const Eigen::Matrix<double, -1, 1> &,
                                             const Eigen::Matrix<double, -1, -1> &,
                                             Eigen::Matrix<double, -1, 1> &,
                                             Eigen::SparseMatrix<double> &,
                                             bool);
    template double compute_energy_from_jacobian<double>(const Eigen::Matrix<double, -1, -1> &, const Eigen::Matrix<double, -1, 1> &);
    
} // namespace wmtk
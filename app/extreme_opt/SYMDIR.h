#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <iostream>


namespace wmtk{
    template <typename T>
    T symmetric_dirichlet_energy_t(T a, T b, T c, T d)
    {
        auto det = a * d - b * c;
        auto frob2 = a * a + b * b + c * c + d * d;
        return frob2 * (1 + 1 / (det * det));
    }

    template <typename Derived>
    inline auto symmetric_dirichlet_energy(const Eigen::MatrixBase<Derived> &a,
                                        const Eigen::MatrixBase<Derived> &b, const Eigen::MatrixBase<Derived> &c, const Eigen::MatrixBase<Derived> &d)
    {
    auto det = a.array() * d.array() - b.array() * c.array();
    auto frob2 = a.array().abs2() + b.array().abs2() + c.array().abs2() + d.array().abs2();
    return (frob2 * (1 + (det).abs2().inverse())).matrix();
    }

    template <typename Scalar>
    Scalar compute_energy_from_jacobian(const Eigen::Matrix<Scalar, -1, -1> &J, const Eigen::Matrix<Scalar, -1, 1> &area);

    template <typename Scalar>
    Scalar get_grad_and_hessian(const Eigen::SparseMatrix<Scalar> &G,
                            const Eigen::Matrix<Scalar, -1, 1> &area,
                            const Eigen::Matrix<Scalar, -1, -1> &uv,
                            Eigen::Matrix<Scalar, -1, 1> &grad,
                            Eigen::SparseMatrix<Scalar> &hessian,
                            bool get_hessian);

    template <typename Scalar>
    void jacobian_from_uv(const Eigen::SparseMatrix<Scalar> &G, const Eigen::Matrix<Scalar, -1, -1> &uv, Eigen::Matrix<Scalar, -1, -1> &Ji);
}

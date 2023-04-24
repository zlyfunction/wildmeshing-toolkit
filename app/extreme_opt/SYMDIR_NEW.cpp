#include "SYMDIR_NEW.h"
#include <igl/grad.h>
#include <igl/cat.h>
#include <igl/local_basis.h>

namespace wmtk{

template <typename Scalar>
Eigen::Matrix2<Scalar> 
SymmetricDirichletEnergy::jacobian(
        const Eigen::Vector<Scalar, 3>& A,
        const Eigen::Vector<Scalar, 3>& B,
        const Eigen::Vector<Scalar, 3>& C,
        const Eigen::Vector<Scalar, 2>& a,
        const Eigen::Vector<Scalar, 2>& b,
        const Eigen::Vector<Scalar, 2>& c)
{

    // igl::grad
    Eigen::Vector<Scalar, 3> v32 = C - B;
    Eigen::Vector<Scalar, 3> v13 = A - C;
    Eigen::Vector<Scalar, 3> v21 = B - A;
    Eigen::Vector<Scalar, 3> n = v32.cross(v13);
    Scalar dblA = sqrt(n.dot(n));
    Eigen::Vector<Scalar, 3> u = n / dblA;

    Scalar norm21 = sqrt(v21.dot(v21));
    Scalar norm13 = sqrt(v13.dot(v13));
    Eigen::Vector<Scalar, 3> eperp21, eperp13;
    eperp21 = u.cross(v21);
    eperp21 = eperp21 / sqrt(eperp21.dot(eperp21));
    eperp21 = eperp21 * (norm21 / dblA);
    eperp13 = u.cross(v13);
    eperp13 = eperp13 / sqrt(eperp13.dot(eperp13));
    eperp13 = eperp13 * (norm13 / dblA);

    Eigen::MatrixX<Scalar> G(3, 3);
    G.setZero();
    for (int d = 0; d < 3; d++)
    {
        G(d, 0) = -eperp13(d) - eperp21(d);
        G(d, 1) = eperp13(d);
        G(d, 2) = eperp21(d);
    }
    
    // igl::local_basis
    Eigen::Vector<Scalar, 3> F1;
    Eigen::Vector<Scalar, 3> F2;
    F1 = v21 / sqrt(v21.dot(v21));
    F2 = eperp21 / sqrt(eperp21.dot(eperp21));

    // get Dx and Dy w.r.t local_basis
    Eigen::Vector<Scalar, 3> Dx = F1.transpose() * G;
    Eigen::Vector<Scalar, 3> Dy = F2.transpose() * G;

    Eigen::Vector<Scalar, 3> us, vs;
    us << a(0), b(0), c(0);
    vs << a(1), b(1), c(1);
    Eigen::Matrix2<Scalar> J;

    J << Dx.dot(us), Dx.dot(vs), Dy.dot(us), Dy.dot(vs);

    return J;
}

template <typename Scalar>
Eigen::Matrix2<Scalar> 
SymmetricDirichletEnergy::jacobian_old(
        const Eigen::Vector<Scalar, 3>& A,
        const Eigen::Vector<Scalar, 3>& B,
        const Eigen::Vector<Scalar, 3>& C,
        const Eigen::Vector<Scalar, 2>& a,
        const Eigen::Vector<Scalar, 2>& b,
        const Eigen::Vector<Scalar, 2>& c)
{
    Eigen::MatrixXi F(1,3);
    F.row(0) << 0, 1, 2;
    Eigen::MatrixX<Scalar> V(3,3);
    V.row(0) = A; V.row(1) = B; V.row(2) = C;
    Eigen::MatrixX<Scalar> uv(3,2);
    uv.row(0) = a; uv.row(1) = b; uv.row(2) = c;

    Eigen::SparseMatrix<Scalar> G_sparse;
    igl::grad(V, F, G_sparse, false);
    Eigen::MatrixX<Scalar> G(G_sparse);

    Eigen::MatrixX<Scalar> F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);

    Eigen::MatrixX<Scalar> Dx = F1 * G;
    Eigen::MatrixX<Scalar> Dy = F2 * G;

    Eigen::MatrixX<Scalar> hstack = igl::cat(1, Dx, Dy);
    Eigen::MatrixX<Scalar> empty(hstack.rows(), hstack.cols());
    Eigen::MatrixX<Scalar> grad_op = igl::cat(1, igl::cat(2, hstack, empty), igl::cat(2, empty, hstack));
    
    Eigen::Matrix<Scalar, -1, 1> altJ = grad_op * Eigen::Map<const Eigen::Matrix<Scalar, -1, 1>>(uv.data(), uv.size());

    Eigen::Matrix2<Scalar> J = Eigen::Map<Eigen::MatrixX<Scalar>>(altJ.data(), 2, 2);

    return J;
}

template Eigen::Matrix2<double> 
SymmetricDirichletEnergy::jacobian_old<double>(
        const Eigen::Vector<double, 3>&,
        const Eigen::Vector<double, 3>&,
        const Eigen::Vector<double, 3>&,
        const Eigen::Vector<double, 2>&,
        const Eigen::Vector<double, 2>&,
        const Eigen::Vector<double, 2>&);

template Eigen::Matrix2<double> 
SymmetricDirichletEnergy::jacobian<double>(
        const Eigen::Vector<double, 3>&,
        const Eigen::Vector<double, 3>&,
        const Eigen::Vector<double, 3>&,
        const Eigen::Vector<double, 2>&,
        const Eigen::Vector<double, 2>&,
        const Eigen::Vector<double, 2>&);

template Eigen::Matrix2<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>> 
SymmetricDirichletEnergy::jacobian<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>>(
        const Eigen::Vector<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>, 3>&,
        const Eigen::Vector<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>, 3>&,
        const Eigen::Vector<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>, 3>&,
        const Eigen::Vector<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>, 2>&,
        const Eigen::Vector<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>, 2>&,
        const Eigen::Vector<DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>>, 2>&);
}

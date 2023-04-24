#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <iostream>
#include "ExtremeOpt.h"
#include "autodiff_jakob.h"
namespace wmtk {


class SymmetricDirichletEnergy
{
public:
    enum class EnergyType { Max, Lp, SoftMax };

    SymmetricDirichletEnergy(EnergyType e_type, int expo = 2)
    {
        energy_type = e_type;
        exponent = expo;
    };

    //===============================================
    // Single triangle interfaces
    //===============================================
    template <typename T>
    T symmetric_dirichlet_energy_from_jacobian(const T& a, const T& b, const T& c, const T& d)
    {
        auto ad = a * d;
        auto bc = b * c;
        auto det = ad - bc;

        auto a2 = a * a;
        auto b2 = b * b;
        auto c2 = c * c;
        auto d2 = d * d;
        auto a2b2 = a2 + b2;
        auto c2d2 = c2 + d2;
        auto frob2 = a2b2 + c2d2;

        return frob2 * (1 + 1 / (det * det));
    }

    template <typename Scalar>
    Eigen::Matrix2<Scalar> jacobian(
        const Eigen::Vector<Scalar, 3>& A,
        const Eigen::Vector<Scalar, 3>& B,
        const Eigen::Vector<Scalar, 3>& C,
        const Eigen::Vector<Scalar, 2>& a,
        const Eigen::Vector<Scalar, 2>& b,
        const Eigen::Vector<Scalar, 2>& c);

    template <typename Scalar>
    Eigen::Matrix2<Scalar> jacobian_old(
        const Eigen::Vector<Scalar, 3>& A,
        const Eigen::Vector<Scalar, 3>& B,
        const Eigen::Vector<Scalar, 3>& C,
        const Eigen::Vector<Scalar, 2>& a,
        const Eigen::Vector<Scalar, 2>& b,
        const Eigen::Vector<Scalar, 2>& c);

    // A = 3d coords
    // a = uv coords
    template <typename Scalar>
    Scalar symmetric_dirichlet_energy(const Eigen::Vector<Scalar, 3>&A, const Eigen::Vector<Scalar, 3>&B, const Eigen::Vector<Scalar, 3>&C, const Eigen::Vector<Scalar, 2>&a, const Eigen::Vector<Scalar, 2>&b, const Eigen::Vector<Scalar, 2>&c)
    {
#define AVOID_ROTATION_ERROR
#ifdef AVOID_ROTATION_ERROR
        Scalar E(0);
        Eigen::Matrix2<Scalar> J = jacobian(A, B, C, a, b, c);
        E = E + symmetric_dirichlet_energy_from_jacobian(J(0,0), J(0,1), J(1,0), J(1,1));
        J = jacobian(C, A, B, c, a, b);
        E = E + symmetric_dirichlet_energy_from_jacobian(J(0,0), J(0,1), J(1,0), J(1,1));
        J = jacobian(B, C, A, b, c, a);
        E = E + symmetric_dirichlet_energy_from_jacobian(J(0,0), J(0,1), J(1,0), J(1,1));
        E = E / 3;
        return E;
#else
        Eigen::Matrix2<Scalar> J = jacobian(A, B, C, a, b, c);
        return symmetric_dirichlet_energy_from_jacobian(J(0,0), J(0,1), J(1,0), J(1,1));
#endif
    }


    //===============================================
    // Multiple triangle interfaces
    //===============================================

    
    double get_grad_and_hessian_onering(extremeopt::ExtremeOpt &m, extremeopt::ExtremeOpt::Tuple t,Eigen::Matrix<double, 2, 1>& grad, Eigen::Matrix2<double>& hessian)
    {
        grad.setZero(); hessian.setZero();
        auto locs = m.get_one_ring_tris_for_vertex(t);
        typedef DScalar2<double, Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>> DScalar;
        DiffScalarBase::setVariableCount(2);
        Eigen::VectorX<DScalar> areas(locs.size());
        Eigen::VectorX<DScalar> energy(locs.size());
        int id_cnt = 0;
        for (auto loc : locs) {
            Eigen::Vector<double, 3> A, B, C;
            Eigen::Vector<double, 2> a, b, c;

            get_ordered_abc(m, t, loc, A, B, C, a, b, c);

            double area = (B-A).cross(C-A).norm();
            
            Eigen::Vector<DScalar, 3> AA, BB, CC;
            Eigen::Vector<DScalar, 2> aa, bb, cc;

            AA << DScalar(A(0)), DScalar(A(1)), DScalar(A(2));
            BB << DScalar(B(0)), DScalar(B(1)), DScalar(B(2));
            CC << DScalar(C(0)), DScalar(C(1)), DScalar(C(2));

            aa << DScalar(0, a(0)), DScalar(1, a(1));
            bb << DScalar(b(0)), DScalar(b(1));
            cc << DScalar(c(0)), DScalar(c(1));

            energy(id_cnt) = symmetric_dirichlet_energy(AA, BB, CC, aa, bb, cc);
            areas(id_cnt) = DScalar(area);
            id_cnt++;
        }

        DScalar E;
        switch (energy_type)
        {
        case EnergyType::Lp:
            E = Lp(areas, energy);
            break;
        case EnergyType::SoftMax:
            E = SoftMax(energy);
            break;
        }

        grad = E.getGradient();
        hessian = E.getHessian();
        return E.getValue();
    }

    double symmetric_dirichlet_energy_onering(extremeopt::ExtremeOpt &m, extremeopt::ExtremeOpt::Tuple t)
    {
        double E = 0;
        auto locs = m.get_one_ring_tris_for_vertex(t);
        Eigen::VectorXd areas(locs.size());
        Eigen::VectorXd energy(locs.size());

        int id_cnt = 0;
        for (auto loc : locs) {
            Eigen::Vector<double, 3> A, B, C;
            Eigen::Vector<double, 2> a, b, c;

            get_ordered_abc(m, t, loc, A, B, C, a, b, c);

            areas(id_cnt) =  (B-A).cross(C-A).norm();
            energy(id_cnt) = symmetric_dirichlet_energy(A, B, C, a, b, c);
            id_cnt++;
        }
        switch (energy_type)
        {
        case EnergyType::Lp:
            E = Lp(areas, energy);
            break;
        case EnergyType::SoftMax:
            E = SoftMax(energy);
            break;
        case EnergyType::Max:
            E = energy.maxCoeff();
            break;
        }
        return E;
    }

    double symmetric_dirichlet_energy_2chart(extremeopt::ExtremeOpt &m, extremeopt::ExtremeOpt::Tuple t)
    {
        double E = 0;
        std::vector<extremeopt::ExtremeOpt::Tuple> locs;
        locs.push_back(t);
        if (t.switch_face(m).has_value())
        {
            locs.push_back(t.switch_face(m).value());
        }
        Eigen::VectorXd areas(locs.size());
        Eigen::VectorXd energy(locs.size());

        int id_cnt = 0;
        for (auto loc : locs) {
            Eigen::Vector<double, 3> A, B, C;
            Eigen::Vector<double, 2> a, b, c;

            get_ordered_abc(m, t, loc, A, B, C, a, b, c);

            areas(id_cnt) =  (B-A).cross(C-A).norm();
            energy(id_cnt) = symmetric_dirichlet_energy(A, B, C, a, b, c);
            id_cnt++;
        }
        switch (energy_type)
        {
        case EnergyType::Lp:
            E = Lp(areas, energy);
            break;
        case EnergyType::SoftMax:
            E = SoftMax(energy);
            break;
        case EnergyType::Max:
            E = energy.maxCoeff();
            break;
        }
        return E;

    }



private:
    EnergyType energy_type = EnergyType::Lp;
    int exponent = 2; // Exponent used for Lp

    template <typename Scalar>
    Scalar Lp(Eigen::VectorX<Scalar> &area, Eigen::VectorX<Scalar> &energy)
    {
        Scalar E(0);
        for (int i = 0; i < energy.size(); i++)
        {
            E = E + area(i) * pow(energy(i), exponent);
        }
        return E;
    }

    template <typename Scalar>
    Scalar SoftMax(Eigen::VectorX<Scalar> &energy)
    {
        Scalar E(0), sum_exp(0);
        for (int i = 0; i < energy.size(); i++)
        {
            sum_exp = sum_exp + exp(energy(i));
        }
        for (int i = 0; i < energy.size(); i++)
        {
            E = E + (exp(energy(i)) / sum_exp) * energy(i);
        }
        return E;
    }

    void get_ordered_abc(extremeopt::ExtremeOpt &m, extremeopt::ExtremeOpt::Tuple &t, extremeopt::ExtremeOpt::Tuple &loc, Eigen::Vector<double, 3> &A, Eigen::Vector<double, 3> &B, Eigen::Vector<double, 3> &C, Eigen::Vector<double, 2> &a, Eigen::Vector<double, 2> &b, Eigen::Vector<double, 2> &c)
    {
        auto local_tuples = m.oriented_tri_vertices(loc);
        int vid_local = 0; 
        for (int i = 0; i < local_tuples.size(); i++)
        {
            if (local_tuples[i].vid(m) == t.vid(m))
            {
                vid_local = i;
                break;
            }
        }
        A = m.vertex_attrs[local_tuples[vid_local].vid(m)].pos_3d;
        B = m.vertex_attrs[local_tuples[(vid_local + 1) % 3].vid(m)].pos_3d;
        C = m.vertex_attrs[local_tuples[(vid_local + 2) % 3].vid(m)].pos_3d;
        a = m.vertex_attrs[local_tuples[vid_local].vid(m)].pos;
        b = m.vertex_attrs[local_tuples[(vid_local + 1) % 3].vid(m)].pos;
        c = m.vertex_attrs[local_tuples[(vid_local + 2) % 3].vid(m)].pos;
    }
};


} // namespace wmtk

#pragma once

#include <CGAL/Gmpq.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/number_utils.h>
#include <gmp.h>
#include <gmpxx.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace wmtk {

class Rational
{
public:
    Rational(bool rounded = false);
    Rational(int v, bool rounded = false);
    Rational(long v, bool rounded = false);
    Rational(double d, bool rounded = false);
    Rational(const Eigen::Matrix<char, Eigen::Dynamic, 1>& data, bool rounded = false);
    Rational(const mpq_t& v_);
    Rational(const Rational& other);
    Rational(const Rational& other, bool rounded);
    Rational(const std::string& data, bool rounded = false);

    // Conversion from CGAL::Gmpq (exact, no precision loss)
    Rational(const CGAL::Gmpq& cgal_rational, bool rounded = false)
    {
        if (rounded) {
            m_is_rounded = true;
            d_value = CGAL::to_double(cgal_rational);
        } else {
            m_is_rounded = false;
            mpq_init(value);
            mpq_set(value, cgal_rational.mpq());
            d_value = std::numeric_limits<double>::lowest();
        }
    }

    // Conversion from CGAL::Lazy_exact_nt<CGAL::Gmpq> (exact, no precision loss)
    template <typename NT>
    Rational(const CGAL::Lazy_exact_nt<NT>& cgal_rational, bool rounded = false)
    {
        if (rounded) {
            m_is_rounded = true;
            d_value = CGAL::to_double(cgal_rational);
        } else {
            m_is_rounded = false;
            mpq_init(value);
            // Use exact() to get the underlying expression (__gmp_expr<mpq_t, mpq_t>)
            // Convert to CGAL::Gmpq by using CGAL::Gmpq constructor that accepts mpq_t
            auto exact_val = cgal_rational.exact();
            // __gmp_expr has an implicit conversion to mpq_t via get_mpq_t()
            // But we need to use it correctly. Let's try using CGAL::Gmpq constructor
            // that accepts mpq_t, and use get_mpq_t() to get the mpq_t from __gmp_expr
            CGAL::Gmpq gmpq_val(exact_val.get_mpq_t());
            mpq_set(value, gmpq_val.mpq());
            d_value = std::numeric_limits<double>::lowest();
        }
    }

    Rational& operator=(const Rational& x);
    Rational& operator=(const double x);

    // Compound assignment operators required by Eigen
    Rational& operator+=(const Rational& x);
    Rational& operator-=(const Rational& x);
    Rational& operator*=(const Rational& x);
    Rational& operator/=(const Rational& x);

    // Square root function required by Eigen
    friend Rational sqrt(const Rational& x);

    template <typename T>
    void init(const T& v)
    {
        mpq_set(value, v);
        m_is_rounded = false;
    }

    ~Rational();

    void canonicalize();

    friend Rational operator+(const Rational& x, const Rational& y);
    friend Rational operator-(const Rational& x, const Rational& y);

    friend Rational operator-(const Rational& x);

    friend Rational pow(const Rational& x, int p);
    friend Rational abs(const Rational& r0);
    int get_sign() const;

    friend Rational operator*(const Rational& x, const Rational& y);
    friend Rational operator/(const Rational& x, const Rational& y);

    //> < ==
    friend bool operator<(const Rational& r, const Rational& r1) { return cmp(r, r1) < 0; }
    friend bool operator>(const Rational& r, const Rational& r1) { return cmp(r, r1) > 0; }
    friend bool operator<=(const Rational& r, const Rational& r1) { return cmp(r, r1) <= 0; }
    friend bool operator>=(const Rational& r, const Rational& r1) { return cmp(r, r1) >= 0; }

    friend bool operator==(const Rational& r, const Rational& r1);
    friend bool operator!=(const Rational& r, const Rational& r1);

    // to double
    double to_double() const;
    explicit operator double() const;

    friend std::ostream& operator<<(std::ostream& os, const Rational& r);

    inline void round()
    {
        if (m_is_rounded) return;

        d_value = this->to_double();
        m_is_rounded = true;
        mpq_clear(value);
    }

    inline bool can_be_rounded()
    {
        if (m_is_rounded) return true;

        return this->to_double() == *this;
    }

    void init_from_binary(const std::string& v);
    std::string to_binary() const;

    std::string serialize() const;
    static Rational deserialize(const std::string& s);

    inline bool is_rounded() const { return m_is_rounded; }

    void export_mpq(mpq_t out) const;

private:
    mpq_t value;
    double d_value;
    bool m_is_rounded;

    friend int cmp(const Rational& r, const Rational& r1);

    std::string numerator() const;
    std::string denominator() const;
};

} // namespace wmtk

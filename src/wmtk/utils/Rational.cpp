#include "Rational.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>

namespace wmtk {
void Rational::canonicalize()
{
    if (m_is_rounded) return;
    mpq_canonicalize(value);
}

int Rational::get_sign() const
{
    if (m_is_rounded) return d_value == 0 ? 0 : (d_value < 0 ? -1 : 1);

    return mpq_sgn(value);
}

Rational::Rational(bool rounded)
    : m_is_rounded(rounded)
{
    if (m_is_rounded) {
        d_value = 0.0;
    } else {
        mpq_init(value);
        mpq_set_d(value, 0.0);
        d_value = std::numeric_limits<double>::lowest();
    }
}

Rational::Rational(int v, bool rounded)
    : m_is_rounded(rounded)
{
    if (m_is_rounded) {
        d_value = (double)v;
    } else {
        mpq_init(value);
        mpq_set_d(value, (double)v);
        d_value = std::numeric_limits<double>::lowest();
    }
}

Rational::Rational(long v, bool rounded)
    : m_is_rounded(rounded)
{
    if (m_is_rounded) {
        d_value = (double)v;
    } else {
        mpq_init(value);
        mpq_set_d(value, (double)v);
        d_value = std::numeric_limits<double>::lowest();
    }
}

Rational::Rational(double d, bool rounded)
    : m_is_rounded(rounded)
{
    if (m_is_rounded) {
        d_value = d;
    } else {
        mpq_init(value);
        mpq_set_d(value, d);
        d_value = std::numeric_limits<double>::lowest();
    }
}

Rational::Rational(const mpq_t& v_)
{
    mpq_init(value);
    mpq_set(value, v_);
    // canonicalize();
    m_is_rounded = false;

    d_value = std::numeric_limits<double>::lowest();
}

Rational::Rational(const Rational& other)
    : d_value(other.d_value)
    , m_is_rounded(other.m_is_rounded)
{
    if (!m_is_rounded) {
        mpq_init(value);
        mpq_set(value, other.value);
    }
}
Rational::Rational(const Rational& other, bool rounded)
    : d_value(other.d_value)
    , m_is_rounded(rounded)
{
    if (!m_is_rounded) {
        mpq_init(value);

        if (other.m_is_rounded)
            mpq_set_d(value, other.d_value);
        else
            mpq_set(value, other.value);

        d_value = std::numeric_limits<double>::lowest();
    } else {
        d_value = other.to_double();
    }
}


Rational::Rational(const std::string& data, bool rounded)
    : m_is_rounded(rounded)
{
    if (m_is_rounded) {
        mpq_t tmp_r;
        mpq_init(tmp_r);
        mpq_set_str(tmp_r, data.c_str(), 10);

        d_value = mpq_get_d(tmp_r);
        mpq_clear(tmp_r);
    } else {
        mpq_init(value);
        mpq_set_str(value, data.c_str(), 10);
        d_value = std::numeric_limits<double>::lowest();
    }
}

Rational::~Rational()
{
    if (!m_is_rounded) mpq_clear(value);
}

Rational operator+(const Rational& x, const Rational& y)
{
    if (x.m_is_rounded && y.m_is_rounded) {
        return Rational(x.d_value + y.d_value, true);
    }

    Rational r_out;

    if (x.m_is_rounded && !y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_add(r_out.value, tmp, y.value);
        mpq_clear(tmp);
    } else if (!x.m_is_rounded && y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, y.d_value);
        mpq_add(r_out.value, x.value, tmp);
        mpq_clear(tmp);
    } else
        mpq_add(r_out.value, x.value, y.value);

    return r_out;
}

Rational operator-(const Rational& x, const Rational& y)
{
    if (x.m_is_rounded && y.m_is_rounded) {
        return Rational(x.d_value - y.d_value, true);
    }

    Rational r_out;

    if (x.m_is_rounded && !y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_sub(r_out.value, tmp, y.value);
        mpq_clear(tmp);
    } else if (!x.m_is_rounded && y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, y.d_value);
        mpq_sub(r_out.value, x.value, tmp);
        mpq_clear(tmp);
    } else
        mpq_sub(r_out.value, x.value, y.value);

    return r_out;
}


Rational operator-(const Rational& x)
{
    if (x.m_is_rounded) return Rational(-x.d_value, true);

    Rational r_out;
    mpq_neg(r_out.value, x.value);
    return r_out;
}

Rational pow(const Rational& x, int p)
{
    if (x.m_is_rounded) return Rational(std::pow(x.d_value, p), true);

    Rational r_out = x;
    for (int i = 1; i < std::abs(p); i++) {
        r_out = r_out * x;
    }
    if (p < 0) return Rational(1.0, false) / r_out;
    return r_out;
}

Rational operator*(const Rational& x, const Rational& y)
{
    if (x.m_is_rounded && y.m_is_rounded) {
        return Rational(x.d_value * y.d_value, true);
    }

    Rational r_out;

    if (x.m_is_rounded && !y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_mul(r_out.value, tmp, y.value);
        mpq_clear(tmp);
    } else if (!x.m_is_rounded && y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, y.d_value);
        mpq_mul(r_out.value, x.value, tmp);
        mpq_clear(tmp);
    } else
        mpq_mul(r_out.value, x.value, y.value);

    return r_out;
}

Rational operator/(const Rational& x, const Rational& y)
{
    if (x.m_is_rounded && y.m_is_rounded) {
        return Rational(x.d_value / y.d_value, true);
    }

    Rational r_out;

    if (x.m_is_rounded && !y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_div(r_out.value, tmp, y.value);
        mpq_clear(tmp);
    } else if (!x.m_is_rounded && y.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, y.d_value);
        mpq_div(r_out.value, x.value, tmp);
        mpq_clear(tmp);
    } else
        mpq_div(r_out.value, x.value, y.value);

    return r_out;
}

Rational& Rational::operator=(const Rational& x)
{
    if (this == &x) return *this;

    if (!m_is_rounded) mpq_clear(value);

    m_is_rounded = x.m_is_rounded;
    d_value = x.d_value;

    if (!x.m_is_rounded) { //&& !m_is_rounded
        mpq_init(value);
        mpq_set(value, x.value);
    }

    return *this;
}

Rational& Rational::operator=(const double x)
{
    if (m_is_rounded)
        d_value = x;
    else {
        mpq_set_d(value, x);
        d_value = std::numeric_limits<double>::lowest();
    }
    //            canonicalize();
    return *this;
}

int cmp(const Rational& r, const Rational& r1)
{
    if (r.m_is_rounded && r1.m_is_rounded)
        return r.d_value == r1.d_value ? 0 : (r.d_value > r1.d_value ? 1 : -1);

    if (r.m_is_rounded && !r1.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, r.d_value);
        int res = mpq_cmp(tmp, r1.value);
        mpq_clear(tmp);
        return res;
    }

    if (!r.m_is_rounded && r1.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, r1.d_value);
        int res = mpq_cmp(r.value, tmp);
        mpq_clear(tmp);
        return res;
    }

    return mpq_cmp(r.value, r1.value);
}


bool operator==(const Rational& r, const Rational& r1)
{
    if (r.m_is_rounded && r1.m_is_rounded) return r.d_value == r1.d_value;

    if (r.m_is_rounded && !r1.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, r.d_value);
        bool res = mpq_equal(tmp, r1.value);
        mpq_clear(tmp);
        return res;
    }

    if (!r.m_is_rounded && r1.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, r1.d_value);
        bool res = mpq_equal(r.value, tmp);
        mpq_clear(tmp);
        return res;
    }

    return mpq_equal(r.value, r1.value);
}

bool operator!=(const Rational& r, const Rational& r1)
{
    return !(r == r1);
}

// to double
double Rational::to_double() const
{
    if (m_is_rounded) return d_value;
    return mpq_get_d(value);
}

Rational::operator double() const
{
    return to_double();
}

Rational abs(const Rational& r0)
{
    if (r0.m_is_rounded) {
        return Rational(std::abs(r0.d_value), true);
    } else {
        Rational r;
        mpq_abs(r.value, r0.value);
        return r;
    }
}

//<<
std::ostream& operator<<(std::ostream& os, const Rational& r)
{
    os << r.to_double();
    return os;
}

void Rational::init_from_binary(const std::string& v)
{
    mpq_set_str(value, v.c_str(), 2);
}
std::string Rational::to_binary() const
{
    if (m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, d_value);

        std::string v(mpq_get_str(NULL, 2, tmp));
        mpq_clear(tmp);
        return v;
    }

    std::string v(mpq_get_str(NULL, 2, value));
    return v;
}

std::string Rational::serialize() const
{
    return numerator() + "/" + denominator() + "/" + (m_is_rounded ? "1" : "0");
}

Rational Rational::deserialize(const std::string& s)
{
    // Parse the serialized string format: "numerator/denominator/is_rounded"
    size_t first_slash = s.find('/');
    size_t second_slash = s.find('/', first_slash + 1);

    if (first_slash == std::string::npos || second_slash == std::string::npos) {
        // Invalid format, return zero
        return Rational(0);
    }

    std::string num_str = s.substr(0, first_slash);
    std::string denom_str = s.substr(first_slash + 1, second_slash - first_slash - 1);
    std::string rounded_str = s.substr(second_slash + 1);

    bool is_rounded = (rounded_str == "1");

    // Create a Rational and set its value using mpq
    Rational result(is_rounded);

    if (!is_rounded) {
        // Set the numerator and denominator directly
        mpz_t num, denom;
        mpz_init(num);
        mpz_init(denom);

        mpz_set_str(num, num_str.c_str(), 10);  // base 10
        mpz_set_str(denom, denom_str.c_str(), 10);

        mpq_set_num(result.value, num);
        mpq_set_den(result.value, denom);
        mpq_canonicalize(result.value);

        mpz_clear(num);
        mpz_clear(denom);
    } else {
        // For rounded values, reconstruct from numerator/denominator
        mpq_t tmp;
        mpq_init(tmp);

        mpz_t num, denom;
        mpz_init(num);
        mpz_init(denom);

        mpz_set_str(num, num_str.c_str(), 10);
        mpz_set_str(denom, denom_str.c_str(), 10);

        mpq_set_num(tmp, num);
        mpq_set_den(tmp, denom);
        mpq_canonicalize(tmp);

        result.d_value = mpq_get_d(tmp);

        mpz_clear(num);
        mpz_clear(denom);
        mpq_clear(tmp);
    }

    return result;
}

void Rational::export_mpq(mpq_t out) const
{
    if (m_is_rounded) {
        mpq_set_d(out, d_value);
    } else {
        mpq_set(out, value);
    }
}

Rational::Rational(const Eigen::Matrix<char, Eigen::Dynamic, 1>& data, bool rounded)
    : m_is_rounded(rounded)
{
    if (m_is_rounded) {
        d_value = 0.0;
    } else {
        mpq_init(value);
        mpq_set_d(value, 0.0);
        d_value = std::numeric_limits<double>::lowest();
    }
}


std::string Rational::numerator() const
{
    mpq_t tmp;
    mpq_init(tmp);
    if (m_is_rounded)
        mpq_set_d(tmp, d_value);
    else
        mpq_set(tmp, value);


    mpz_t num;
    mpz_init(num);

    mpq_get_num(num, tmp);
    std::string v(mpz_get_str(NULL, 10, num));

    mpz_clear(num);
    mpq_clear(tmp);

    return v;
}

std::string Rational::denominator() const
{
    mpq_t tmp;
    mpq_init(tmp);
    if (m_is_rounded)
        mpq_set_d(tmp, d_value);
    else
        mpq_set(tmp, value);

    mpz_t denom;
    mpz_init(denom);
    mpq_get_den(denom, tmp);

    std::string v(mpz_get_str(NULL, 10, denom));

    mpz_clear(denom);
    mpq_clear(tmp);

    return v;
}

// Compound assignment operators
Rational& Rational::operator+=(const Rational& x)
{
    if (m_is_rounded && x.m_is_rounded) {
        d_value += x.d_value;
        return *this;
    }

    if (m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, d_value);
        mpq_add(tmp, tmp, x.value);
        d_value = mpq_get_d(tmp);
        mpq_clear(tmp);
        return *this;
    }

    if (x.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_add(value, value, tmp);
        mpq_clear(tmp);
        return *this;
    }

    mpq_add(value, value, x.value);
    return *this;
}

Rational& Rational::operator-=(const Rational& x)
{
    if (m_is_rounded && x.m_is_rounded) {
        d_value -= x.d_value;
        return *this;
    }

    if (m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, d_value);
        mpq_sub(tmp, tmp, x.value);
        d_value = mpq_get_d(tmp);
        mpq_clear(tmp);
        return *this;
    }

    if (x.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_sub(value, value, tmp);
        mpq_clear(tmp);
        return *this;
    }

    mpq_sub(value, value, x.value);
    return *this;
}

Rational& Rational::operator*=(const Rational& x)
{
    if (m_is_rounded && x.m_is_rounded) {
        d_value *= x.d_value;
        return *this;
    }

    if (m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, d_value);
        mpq_mul(tmp, tmp, x.value);
        d_value = mpq_get_d(tmp);
        mpq_clear(tmp);
        return *this;
    }

    if (x.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_mul(value, value, tmp);
        mpq_clear(tmp);
        return *this;
    }

    mpq_mul(value, value, x.value);
    return *this;
}

Rational& Rational::operator/=(const Rational& x)
{
    if (m_is_rounded && x.m_is_rounded) {
        d_value /= x.d_value;
        return *this;
    }

    if (m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, d_value);
        mpq_div(tmp, tmp, x.value);
        d_value = mpq_get_d(tmp);
        mpq_clear(tmp);
        return *this;
    }

    if (x.m_is_rounded) {
        mpq_t tmp;
        mpq_init(tmp);
        mpq_set_d(tmp, x.d_value);
        mpq_div(value, value, tmp);
        mpq_clear(tmp);
        return *this;
    }

    mpq_div(value, value, x.value);
    return *this;
}

// Square root function
Rational sqrt(const Rational& x)
{
    if (x.m_is_rounded) {
        return Rational(std::sqrt(x.d_value), true);
    }

    // For exact arithmetic, we need to compute the square root
    // This is an approximation since exact square root of rationals may not be rational
    double approx = std::sqrt(x.to_double());
    return Rational(approx, true);
}

} // namespace wmtk

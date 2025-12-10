#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>

// igl
#include <igl/parallel_for.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <wmtk/utils/Rational.hpp>

// Data structures for query points, curves, and surfaces in tetrahedral meshes

template <typename CoordType = double>
struct query_point_tet_t
{
    int64_t t_id; // tet id
    Eigen::Matrix<CoordType, 4, 1> bc; // barycentric coordinates
    Eigen::Vector4i tv_ids; // tet vertex ids
};

// Type aliases for convenience
using query_point_tet = query_point_tet_t<double>;
using query_point_tet_r = query_point_tet_t<wmtk::Rational>;

namespace tet_tracking_utils {
template <typename T>
inline double to_double_scalar(const T& value)
{
    if constexpr (std::is_same_v<T, wmtk::Rational>) {
        return value.to_double();
    } else {
        return static_cast<double>(value);
    }
}

template <typename Scalar, int Rows>
inline Eigen::Matrix<double, Rows, 1> to_double_vector(const Eigen::Matrix<Scalar, Rows, 1>& vec)
{
    Eigen::Matrix<double, Rows, 1> result;
    result.resize(vec.rows());
    for (Eigen::Index i = 0; i < vec.rows(); ++i) {
        result(i) = to_double_scalar(vec(i));
    }
    return result;
}

template <typename CoordType>
inline CoordType clamp_coord(double value)
{
    double clamped = std::max(0.0, std::min(1.0, value));
    if constexpr (std::is_same_v<CoordType, wmtk::Rational>) {
        return wmtk::Rational(clamped);
    } else {
        return static_cast<CoordType>(clamped);
    }
}

template <typename ToType, typename FromType>
inline query_point_tet_t<ToType> convert_query_point_tet(
    const query_point_tet_t<FromType>& from,
    bool normalize = false)
{
    query_point_tet_t<ToType> to;
    to.t_id = from.t_id;
    to.tv_ids = from.tv_ids;
    for (int i = 0; i < 4; ++i) {
        if constexpr (std::is_same_v<FromType, ToType>) {
            to.bc(i) = from.bc(i);
        } else if constexpr (
            std::is_same_v<FromType, wmtk::Rational> && std::is_same_v<ToType, double>) {
            to.bc(i) = from.bc(i).to_double();
        } else if constexpr (
            std::is_same_v<FromType, double> && std::is_same_v<ToType, wmtk::Rational>) {
            to.bc(i) = wmtk::Rational(from.bc(i));
        } else {
            to.bc(i) = static_cast<ToType>(from.bc(i));
        }
    }

    if (normalize) {
        ToType sum = ToType(0);
        for (int i = 0; i < 4; ++i) {
            sum += to.bc(i);
        }
        if (sum != ToType(0)) {
            for (int i = 0; i < 4; ++i) {
                to.bc(i) /= sum;
            }
        }
    }

    return to;
}

inline query_point_tet convert_query_point_tet_to_double(
    const query_point_tet_r& from,
    bool normalize = false)
{
    return convert_query_point_tet<double>(from, normalize);
}

inline query_point_tet_r convert_query_point_tet_to_rational(
    const query_point_tet& from,
    bool normalize = false)
{
    return convert_query_point_tet<wmtk::Rational>(from, normalize);
}
} // namespace tet_tracking_utils

template <typename CoordType>
inline std::ostream& operator<<(std::ostream& os, const query_point_tet_t<CoordType>& qp)
{
    os << "query_point_tet(t_id=" << qp.t_id << ", bc=(";
    for (int i = 0; i < 4; ++i) {
        os << tet_tracking_utils::to_double_scalar(qp.bc(i));
        if (i < 3) {
            os << ",";
        }
    }
    os << "), tv_ids=(" << qp.tv_ids[0] << "," << qp.tv_ids[1] << "," << qp.tv_ids[2] << ","
       << qp.tv_ids[3] << "))";
    return os;
}

struct query_segment_tet
{
    int64_t t_id; // tet id
    Eigen::Vector4d bcs[2]; // barycentric coordinates
    Eigen::Vector4i tv_ids; // tet vertex ids
};

struct query_curve_tet
{
    std::vector<query_segment_tet> segments;
    std::vector<int> next_segment_ids;
};

struct query_triangle_tet
{
    int64_t t_id; // tet id
    Eigen::Vector4d bcs[3]; // barycentric coordinates
    Eigen::Vector4i tv_ids; // tet vertex ids
};

struct query_surface_tet
{
    std::vector<query_triangle_tet> triangles;
};

// new query_surface_tet structure,
struct query_surface_tet_with_connectivity
{
    std::vector<query_point_tet_r> points;
    std::vector<Eigen::Vector3i> query_triangles;
    std::vector<int> tet_ids;
};

// Shared utility functions for barycentric coordinate conversions
template <typename Scalar = double>
Eigen::Matrix<Scalar, 3, 1> barycentric_to_world_tet(
    const Eigen::Matrix<Scalar, 4, 1>& bc,
    const Eigen::Matrix<Scalar, 4, 3>& v);

template <typename Scalar = double>
Eigen::Matrix<Scalar, 4, 1> world_to_barycentric_tet(
    const Eigen::Matrix<Scalar, 3, 1>& p,
    const Eigen::Matrix<Scalar, 4, 3>& v);

// Explicit instantiations
extern template Eigen::Matrix<double, 3, 1> barycentric_to_world_tet<double>(
    const Eigen::Matrix<double, 4, 1>&,
    const Eigen::Matrix<double, 4, 3>&);
extern template Eigen::Matrix<wmtk::Rational, 3, 1> barycentric_to_world_tet<wmtk::Rational>(
    const Eigen::Matrix<wmtk::Rational, 4, 1>&,
    const Eigen::Matrix<wmtk::Rational, 4, 3>&);

extern template Eigen::Matrix<double, 4, 1> world_to_barycentric_tet<double>(
    const Eigen::Matrix<double, 3, 1>&,
    const Eigen::Matrix<double, 4, 3>&);
extern template Eigen::Matrix<wmtk::Rational, 4, 1> world_to_barycentric_tet<wmtk::Rational>(
    const Eigen::Matrix<wmtk::Rational, 3, 1>&,
    const Eigen::Matrix<wmtk::Rational, 4, 3>&);

// Shared file parsing functions
void parse_consolidate_file_tet(
    const json& operation_log,
    std::vector<int64_t>& tet_ids_maps,
    std::vector<int64_t>& vertex_ids_maps);

void parse_non_collapse_file_tet(
    const json& operation_log,
    Eigen::MatrixXd& V_before,
    Eigen::MatrixXi& T_before,
    std::vector<int64_t>& id_map_before,
    std::vector<int64_t>& v_id_map_before,
    Eigen::MatrixXd& V_after,
    Eigen::MatrixXi& T_after,
    std::vector<int64_t>& id_map_after,
    std::vector<int64_t>& v_id_map_after,
    int operation_id = -1);

// Note: Point, curve, and surface tracking functions have been moved to:
// - tet_point_tracking.hpp/cpp
// - tet_curve_tracking.hpp/cpp
// - tet_surface_tracking.hpp/cpp

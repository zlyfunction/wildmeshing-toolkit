#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>
#include "tet_track_operations.hpp"

namespace tet_point_tracking {

// Point tracking core functions
template <typename CoordType = double>
void handle_consolidate_tet(
    const std::vector<int64_t>& tet_ids_maps,
    const std::vector<int64_t>& vertex_ids_maps,
    std::vector<query_point_tet_t<CoordType>>& query_points,
    bool forward);

extern template void handle_consolidate_tet<double>(
    const std::vector<int64_t>&,
    const std::vector<int64_t>&,
    std::vector<query_point_tet_t<double>>&,
    bool);
extern template void handle_consolidate_tet<wmtk::Rational>(
    const std::vector<int64_t>&,
    const std::vector<int64_t>&,
    std::vector<query_point_tet_t<wmtk::Rational>>&,
    bool);

template <typename CoordType = double>
void handle_local_mapping_tet(
    const Eigen::MatrixXd& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixXd& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    std::vector<query_point_tet_t<CoordType>>& query_points);

extern template void handle_local_mapping_tet<double>(
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const std::vector<int64_t>&,
    const std::vector<int64_t>&,
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const std::vector<int64_t>&,
    const std::vector<int64_t>&,
    std::vector<query_point_tet_t<double>>&);
extern template void handle_local_mapping_tet<wmtk::Rational>(
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const std::vector<int64_t>&,
    const std::vector<int64_t>&,
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const std::vector<int64_t>&,
    const std::vector<int64_t>&,
    std::vector<query_point_tet_t<wmtk::Rational>>&);

void handle_local_mapping_tet_exact(
    const Eigen::MatrixX<wmtk::Rational>& V_before,
    const Eigen::MatrixXi& T_before,
    const std::vector<int64_t>& id_map_before,
    const std::vector<int64_t>& v_id_map_before,
    const Eigen::MatrixX<wmtk::Rational>& V_after,
    const Eigen::MatrixXi& T_after,
    const std::vector<int64_t>& id_map_after,
    const std::vector<int64_t>& v_id_map_after,
    std::vector<query_point_tet_t<wmtk::Rational>>& query_points);


template <typename CoordType = double>
void track_point_one_operation_tet(
    const nlohmann::json& operation_log,
    std::vector<query_point_tet_t<CoordType>>& query_points,
    bool do_forward,
    bool use_rational,
    int operation_id);

extern template void track_point_one_operation_tet<double>(
    const nlohmann::json&,
    std::vector<query_point_tet_t<double>>&,
    bool,
    bool,
    int);
extern template void track_point_one_operation_tet<wmtk::Rational>(
    const nlohmann::json&,
    std::vector<query_point_tet_t<wmtk::Rational>>&,
    bool,
    bool,
    int);

template <typename CoordType = double>
void track_point_tet(
    const std::filesystem::path& dirPath,
    std::vector<query_point_tet_t<CoordType>>& query_points,
    bool do_forward,
    bool use_rational);

extern template void track_point_tet<double>(
    const std::filesystem::path&,
    std::vector<query_point_tet_t<double>>&,
    bool,
    bool);
extern template void track_point_tet<wmtk::Rational>(
    const std::filesystem::path&,
    std::vector<query_point_tet_t<wmtk::Rational>>&,
    bool,
    bool);

} // namespace tet_point_tracking

#pragma once

#include <wmtk/utils/Rational.hpp>
#include "cgal_autorefine_sampling.hpp"

namespace cgal_autorefine_demo {

// Rational versions of sampling functions
inline Eigen::Matrix<wmtk::Rational, 4, 1> random_barycentric_rational(std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> type_dist(0, 3);
    std::uniform_int_distribution<int> index_dist(0, 3);
    Eigen::Matrix<wmtk::Rational, 4, 1> weights = Eigen::Matrix<wmtk::Rational, 4, 1>::Zero();
    int sample_type = type_dist(rng);
    if (sample_type == 0) {
        int vertex_idx = index_dist(rng);
        weights[vertex_idx] = wmtk::Rational(1, false);
    } else if (sample_type == 1) {
        std::uniform_int_distribution<int> edge_dist(0, 5);
        int edge_id = edge_dist(rng);
        int edge_v0, edge_v1;
        if (edge_id == 0) {
            edge_v0 = 0;
            edge_v1 = 1;
        } else if (edge_id == 1) {
            edge_v0 = 0;
            edge_v1 = 2;
        } else if (edge_id == 2) {
            edge_v0 = 0;
            edge_v1 = 3;
        } else if (edge_id == 3) {
            edge_v0 = 1;
            edge_v1 = 2;
        } else if (edge_id == 4) {
            edge_v0 = 1;
            edge_v1 = 3;
        } else {
            edge_v0 = 2;
            edge_v1 = 3;
        }
        double t = dist(rng);
        weights[edge_v0] = wmtk::Rational(t, false);
        weights[edge_v1] = wmtk::Rational(1.0 - t, false);
    } else if (sample_type == 2) {
        int excluded_vertex = index_dist(rng);
        int face_v0, face_v1, face_v2;
        if (excluded_vertex == 0) {
            face_v0 = 1;
            face_v1 = 2;
            face_v2 = 3;
        } else if (excluded_vertex == 1) {
            face_v0 = 0;
            face_v1 = 2;
            face_v2 = 3;
        } else if (excluded_vertex == 2) {
            face_v0 = 0;
            face_v1 = 1;
            face_v2 = 3;
        } else {
            face_v0 = 0;
            face_v1 = 1;
            face_v2 = 2;
        }
        double u = dist(rng);
        double v = dist(rng);
        if (u + v > 1.0) {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        weights[face_v0] = wmtk::Rational(u, false);
        weights[face_v1] = wmtk::Rational(v, false);
        weights[face_v2] = wmtk::Rational(1.0 - u - v, false);
    } else {
        Eigen::Vector4d temp_weights;
        do {
            for (int i = 0; i < 4; ++i) {
                temp_weights[i] = dist(rng);
            }
        } while (temp_weights.sum() == 0.0);
        temp_weights /= temp_weights.sum();
        for (int i = 0; i < 4; ++i) {
            weights[i] = wmtk::Rational(temp_weights[i], false);
        }
    }
    return weights;
}

inline bool triangles_intersect_rational(
    const Eigen::Matrix<wmtk::Rational, 3, 1>& p0,
    const Eigen::Matrix<wmtk::Rational, 3, 1>& p1,
    const Eigen::Matrix<wmtk::Rational, 3, 1>& p2,
    const Eigen::Matrix<wmtk::Rational, 3, 1>& p3,
    double eps = 1e-10)
{
    Eigen::Matrix<wmtk::Rational, 3, 1> edge = p2 - p1;
    wmtk::Rational edge_norm_sq = edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2];
    if (edge_norm_sq.to_double() < eps * eps) {
        return false;
    }
    Eigen::Matrix<wmtk::Rational, 3, 1> v0 = p0 - p1;
    Eigen::Matrix<wmtk::Rational, 3, 1> v3 = p3 - p1;
    Eigen::Matrix<wmtk::Rational, 3, 1> cross0;
    cross0[0] = edge[1] * v0[2] - edge[2] * v0[1];
    cross0[1] = edge[2] * v0[0] - edge[0] * v0[2];
    cross0[2] = edge[0] * v0[1] - edge[1] * v0[0];
    Eigen::Matrix<wmtk::Rational, 3, 1> cross3;
    cross3[0] = edge[1] * v3[2] - edge[2] * v3[1];
    cross3[1] = edge[2] * v3[0] - edge[0] * v3[2];
    cross3[2] = edge[0] * v3[1] - edge[1] * v3[0];
    wmtk::Rational dot_product =
        cross0[0] * cross3[0] + cross0[1] * cross3[1] + cross0[2] * cross3[2];
    double dot_val = dot_product.to_double();
    if (dot_val < -eps) {
        return false;
    }
    if (dot_val > eps) {
        return true;
    }
    wmtk::Rational cross0_norm_sq =
        cross0[0] * cross0[0] + cross0[1] * cross0[1] + cross0[2] * cross0[2];
    wmtk::Rational cross3_norm_sq =
        cross3[0] * cross3[0] + cross3[1] * cross3[1] + cross3[2] * cross3[2];
    double cross0_norm = std::sqrt(cross0_norm_sq.to_double());
    double cross3_norm = std::sqrt(cross3_norm_sq.to_double());
    if (cross0_norm < eps && cross3_norm < eps) {
        wmtk::Rational t0 = (v0[0] * edge[0] + v0[1] * edge[1] + v0[2] * edge[2]) / edge_norm_sq;
        wmtk::Rational t3 = (v3[0] * edge[0] + v3[1] * edge[1] + v3[2] * edge[2]) / edge_norm_sq;
        double t0_val = t0.to_double();
        double t3_val = t3.to_double();
        if (t0_val >= 0.0 && t0_val <= 1.0 && t3_val >= 0.0 && t3_val <= 1.0) {
            double overlap = std::min(t0_val, t3_val) + 1.0 - std::max(t0_val, t3_val);
            if (overlap > eps) {
                return true;
            }
        }
    }
    return false;
}

inline Eigen::MatrixXi build_sampled_triangles_rational(
    const Eigen::MatrixXi& T,
    const Eigen::Matrix<wmtk::Rational, Eigen::Dynamic, 3>& V,
    std::vector<Eigen::Matrix<wmtk::Rational, 4, 1>>& sampled_barycentrics,
    std::vector<Eigen::Index>& sampled_tet_indices,
    int triangle_count)
{
    if (triangle_count <= 0) {
        throw std::runtime_error("triangle_count must be positive.");
    }
    if (triangle_count == 1) {
        triangle_count = 2;
    }
    if (triangle_count != 2) {
        std::cerr << "Warning: forcing triangle_count to 2 to keep sampled strip manifold. "
                  << "Requested " << triangle_count << "; generating 2 triangles instead.\n";
    }
    if (T.rows() == 0) {
        throw std::runtime_error("Cannot sample from empty tetrahedral mesh.");
    }
    std::mt19937 rng(1337);
    std::uniform_int_distribution<Eigen::Index> tet_dist(0, T.rows() - 1);
    sampled_barycentrics.clear();
    sampled_tet_indices.clear();
    sampled_barycentrics.reserve(4);
    sampled_tet_indices.reserve(4);
    const std::size_t num_points_needed = 4;
    bool valid_sample = false;
    int max_attempts = 1000;
    int attempt = 0;
    while (!valid_sample && attempt < max_attempts) {
        sampled_barycentrics.clear();
        sampled_tet_indices.clear();
        for (std::size_t i = 0; i < num_points_needed; ++i) {
            Eigen::Index tet_id = tet_dist(rng);
            sampled_tet_indices.push_back(tet_id);
            sampled_barycentrics.push_back(random_barycentric_rational(rng));
        }
        bool all_same_tet = true;
        if (!sampled_tet_indices.empty()) {
            const Eigen::Index first_tet = sampled_tet_indices[0];
            for (std::size_t i = 1; i < sampled_tet_indices.size(); ++i) {
                if (sampled_tet_indices[i] != first_tet) {
                    all_same_tet = false;
                    break;
                }
            }
        }
        if (all_same_tet) {
            ++attempt;
            continue;
        }
        std::vector<Eigen::Matrix<wmtk::Rational, 3, 1>> positions(4);
        for (std::size_t i = 0; i < 4; ++i) {
            Eigen::Index tet_id = sampled_tet_indices[i];
            const Eigen::Matrix<wmtk::Rational, 4, 1>& bc = sampled_barycentrics[i];
            Eigen::Matrix<wmtk::Rational, 3, 1> pos = Eigen::Matrix<wmtk::Rational, 3, 1>::Zero();
            for (int j = 0; j < 4; ++j) {
                pos += bc[j] * V.row(T(tet_id, j)).transpose();
            }
            positions[i] = pos;
        }
        bool intersects =
            triangles_intersect_rational(positions[0], positions[1], positions[2], positions[3]);
        if (!intersects) {
            valid_sample = true;
        } else {
            ++attempt;
        }
    }
    if (!valid_sample) {
        throw std::runtime_error(
            "Failed to sample non-self-intersecting triangles after " +
            std::to_string(max_attempts) + " attempts.");
    }
    Eigen::MatrixXi F(2, 3);
    F.row(0) << 0, 1, 2;
    F.row(1) << 2, 1, 3;
    return F;
}

} // namespace cgal_autorefine_demo

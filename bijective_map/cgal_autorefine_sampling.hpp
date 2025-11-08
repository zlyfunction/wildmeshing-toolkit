#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace cgal_autorefine_demo {

// Shared sampling functions for both double and rational demos
// These functions generate the same sampling pattern using the same RNG seed

inline Eigen::Vector4d random_barycentric(std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Eigen::Vector4d weights;
    do {
        for (int i = 0; i < 4; ++i) {
            weights[i] = dist(rng);
        }
    } while (weights.sum() == 0.0);

    weights /= weights.sum();
    // Bias towards interior by lifting weights slightly
    constexpr double min_weight = 0.05;
    for (int i = 0; i < 4; ++i) {
        weights[i] = std::max(weights[i], min_weight);
    }
    weights /= weights.sum();
    return weights;
}

inline Eigen::MatrixXi build_sampled_triangles(
    const Eigen::MatrixXi& T,
    std::vector<Eigen::Vector4d>& sampled_barycentrics,
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

    std::mt19937 rng(1337); // Fixed seed for reproducibility
    std::uniform_int_distribution<Eigen::Index> tet_dist(0, T.rows() - 1);

    sampled_barycentrics.clear();
    sampled_tet_indices.clear();
    sampled_barycentrics.reserve(4);
    sampled_tet_indices.reserve(4);

    // Sample 4 points randomly, ensuring not all from the same tet
    const std::size_t num_points_needed = 4;
    bool all_same_tet = true;
    int max_attempts = 100;
    int attempt = 0;

    while (all_same_tet && attempt < max_attempts) {
        sampled_barycentrics.clear();
        sampled_tet_indices.clear();

        for (std::size_t i = 0; i < num_points_needed; ++i) {
            Eigen::Index tet_id = tet_dist(rng);
            sampled_tet_indices.push_back(tet_id);
            sampled_barycentrics.push_back(random_barycentric(rng));
        }

        // Check if all points are from the same tet
        all_same_tet = true;
        if (!sampled_tet_indices.empty()) {
            const Eigen::Index first_tet = sampled_tet_indices[0];
            for (std::size_t i = 1; i < sampled_tet_indices.size(); ++i) {
                if (sampled_tet_indices[i] != first_tet) {
                    all_same_tet = false;
                    break;
                }
            }
        }

        ++attempt;
    }

    if (all_same_tet) {
        throw std::runtime_error(
            "Failed to sample points from different tets after " + std::to_string(max_attempts) +
            " attempts. Mesh may have only 1 tet.");
    }

    Eigen::MatrixXi F(2, 3);
    F.row(0) << 0, 1, 2;
    F.row(1) << 2, 1, 3;
    return F;
}

} // namespace cgal_autorefine_demo

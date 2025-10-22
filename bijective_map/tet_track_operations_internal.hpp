#pragma once

#include "tet_track_operations.hpp"
#include <map>
#include <tuple>

// Internal helper functions shared across point/curve/surface tracking modules
//
// Note: This file declares ONLY internal utility functions that are not exposed
// in the public API. Common functions like barycentric_to_world_tet() and
// parse_*_file_tet() are already declared in tet_track_operations.hpp.

// ============================================================================
// Template Functions (must be defined in header)
// ============================================================================

// JSON to matrix conversion template
template <typename Matrix>
Matrix json_to_matrix(const json& js)
{
    int rows = js["rows"];
    int cols = js["values"][0].size();

    Matrix mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = js["values"][i][j];
        }
    }
    return mat;
}

// ============================================================================
// Internal Utility Functions (implementation in tet_track_operations.cpp)
// ============================================================================

// Compute boundary faces of a tetrahedral mesh
// Returns a map from face (sorted vertex triple) to occurrence count
std::map<std::tuple<int, int, int>, int> computeBoundaryFaces(const Eigen::MatrixXi& T);

// Check if a vertex is on the mesh boundary
bool isVertexOnBoundary(
    int vertex_id,
    const std::map<std::tuple<int, int, int>, int>& face_count);

// Find vertex index by exact coordinate matching
int findVertexByCoordinates(const Eigen::Vector3d& point, const Eigen::MatrixXd& V);

// Check if a point on the boundary is coplanar with connected boundary faces
bool checkBoundaryCoplanarity(
    const Eigen::Vector3d& test_point,
    const Eigen::MatrixXd& V,
    const std::map<std::tuple<int, int, int>, int>& face_count);

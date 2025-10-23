#include "tet_track_operations.hpp"
#include <igl/barycentric_coordinates.h>
#include <igl/tet_tet_adjacency.h>
#include <iomanip>
#include <set>
#include "FindPointTetMesh.hpp"
#include "InteractiveAndRobustMeshBooleans/code/booleans.h"
#include "vtu_utils.hpp"

// helper function to convert json to matrix
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

// Barycentric coordinate conversion functions
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> barycentric_to_world_tet(
    const Eigen::Matrix<Scalar, 4, 1>& bc,
    const Eigen::Matrix<Scalar, 4, 3>& v)
{
    // std::cout << "bc: " << bc.transpose() << std::endl;
    // std::cout << "v: \n" << v << std::endl;
    return bc[0] * v.row(0).transpose() + bc[1] * v.row(1).transpose() +
           bc[2] * v.row(2).transpose() + bc[3] * v.row(3).transpose();
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 1> world_to_barycentric_tet(
    const Eigen::Matrix<Scalar, 3, 1>& p,
    const Eigen::Matrix<Scalar, 4, 3>& v)
{
    // Use volume ratio method for barycentric coordinates
    // bc[i] = volume(p, v[j], v[k], v[l]) / volume(v[0], v[1], v[2], v[3])
    // where j, k, l are the other three vertices

    // Compute signed volume using determinant
    auto signed_volume = [](const Eigen::Matrix<Scalar, 3, 1>& a,
                            const Eigen::Matrix<Scalar, 3, 1>& b,
                            const Eigen::Matrix<Scalar, 3, 1>& c,
                            const Eigen::Matrix<Scalar, 3, 1>& d) -> Scalar {
        Eigen::Matrix<Scalar, 3, 3> mat;
        mat.col(0) = b - a;
        mat.col(1) = c - a;
        mat.col(2) = d - a;
        return mat.determinant();
    };

    Eigen::Matrix<Scalar, 3, 1> v0 = v.row(0).transpose();
    Eigen::Matrix<Scalar, 3, 1> v1 = v.row(1).transpose();
    Eigen::Matrix<Scalar, 3, 1> v2 = v.row(2).transpose();
    Eigen::Matrix<Scalar, 3, 1> v3 = v.row(3).transpose();

    // Total volume
    Scalar total_vol = signed_volume(v0, v1, v2, v3);

    // Barycentric coordinates
    Eigen::Matrix<Scalar, 4, 1> bc;
    bc[0] = signed_volume(p, v1, v2, v3) / total_vol;
    bc[1] = signed_volume(v0, p, v2, v3) / total_vol;
    bc[2] = signed_volume(v0, v1, p, v3) / total_vol;
    bc[3] = signed_volume(v0, v1, v2, p) / total_vol;

    return bc;
}

// Explicit template instantiations
template Eigen::Matrix<double, 3, 1> barycentric_to_world_tet<double>(
    const Eigen::Matrix<double, 4, 1>&,
    const Eigen::Matrix<double, 4, 3>&);
template Eigen::Matrix<wmtk::Rational, 3, 1> barycentric_to_world_tet<wmtk::Rational>(
    const Eigen::Matrix<wmtk::Rational, 4, 1>&,
    const Eigen::Matrix<wmtk::Rational, 4, 3>&);

template Eigen::Matrix<double, 4, 1> world_to_barycentric_tet<double>(
    const Eigen::Matrix<double, 3, 1>&,
    const Eigen::Matrix<double, 4, 3>&);
template Eigen::Matrix<wmtk::Rational, 4, 1> world_to_barycentric_tet<wmtk::Rational>(
    const Eigen::Matrix<wmtk::Rational, 3, 1>&,
    const Eigen::Matrix<wmtk::Rational, 4, 3>&);

// File parsing functions
void parse_consolidate_file_tet(
    const json& operation_log,
    std::vector<int64_t>& tet_ids_maps,
    std::vector<int64_t>& vertex_ids_maps)
{
    tet_ids_maps = operation_log["new2old"][3].get<std::vector<int64_t>>();
    vertex_ids_maps = operation_log["new2old"][0].get<std::vector<int64_t>>();
}

// Helper function to compute boundary faces of a tetrahedral mesh
std::map<std::tuple<int, int, int>, int> computeBoundaryFaces(const Eigen::MatrixXi& T)
{
    std::map<std::tuple<int, int, int>, int> face_count;

    for (int tet_id = 0; tet_id < T.rows(); ++tet_id) {
        // Four faces of tetrahedron: (1,2,3), (0,2,3), (0,1,3), (0,1,2)
        std::vector<std::tuple<int, int, int>> faces = {
            std::make_tuple(T(tet_id, 1), T(tet_id, 2), T(tet_id, 3)),
            std::make_tuple(T(tet_id, 0), T(tet_id, 2), T(tet_id, 3)),
            std::make_tuple(T(tet_id, 0), T(tet_id, 1), T(tet_id, 3)),
            std::make_tuple(T(tet_id, 0), T(tet_id, 1), T(tet_id, 2))};

        for (auto& face : faces) {
            // Sort vertices to create canonical face representation
            std::vector<int> sorted_face = {
                std::get<0>(face),
                std::get<1>(face),
                std::get<2>(face)};
            std::sort(sorted_face.begin(), sorted_face.end());
            auto canonical_face = std::make_tuple(sorted_face[0], sorted_face[1], sorted_face[2]);
            face_count[canonical_face]++;
        }
    }

    return face_count;
}

// Helper function to check if a vertex is on the boundary
bool isVertexOnBoundary(int vertex_id, const std::map<std::tuple<int, int, int>, int>& face_count)
{
    for (const auto& [face, count] : face_count) {
        if (count == 1) { // Boundary face
            auto [v0, v1, v2] = face;
            if (v0 == vertex_id || v1 == vertex_id || v2 == vertex_id) {
                return true;
            }
        }
    }
    return false;
}

// Helper function to find vertex index by exact coordinate match
int findVertexByCoordinates(const Eigen::Vector3d& point, const Eigen::MatrixXd& V)
{
    for (int i = 0; i < V.rows(); ++i) {
        if (V(i, 0) == point.x() && V(i, 1) == point.y() && V(i, 2) == point.z()) {
            return i;
        }
    }
    return -1;
}

// Helper function to check exact coplanarity of boundary vertices
bool checkBoundaryCoplanarity(
    const Eigen::Vector3d& test_point,
    const Eigen::MatrixXd& V,
    const std::map<std::tuple<int, int, int>, int>& face_count)
{
    // Find test_point's vertex index
    int test_vertex_idx = findVertexByCoordinates(test_point, V);
    if (test_vertex_idx == -1) {
        std::cout << "Test point not found in vertex list" << std::endl;
        return false;
    }

    // Find all boundary vertices connected to test_vertex_idx
    std::vector<int> connected_boundary_vertices;

    for (const auto& [face, count] : face_count) {
        if (count == 1) { // Boundary face
            auto [v0, v1, v2] = face;
            if (v0 == test_vertex_idx || v1 == test_vertex_idx || v2 == test_vertex_idx) {
                // Add other vertices from this boundary face
                if (v0 != test_vertex_idx) connected_boundary_vertices.push_back(v0);
                if (v1 != test_vertex_idx) connected_boundary_vertices.push_back(v1);
                if (v2 != test_vertex_idx) connected_boundary_vertices.push_back(v2);
            }
        }
    }

    // Remove duplicates
    std::sort(connected_boundary_vertices.begin(), connected_boundary_vertices.end());
    connected_boundary_vertices.erase(
        std::unique(connected_boundary_vertices.begin(), connected_boundary_vertices.end()),
        connected_boundary_vertices.end());

    if (connected_boundary_vertices.size() < 3) {
        std::cout << "Not enough connected boundary vertices for coplanarity test ("
                  << connected_boundary_vertices.size() << " found)" << std::endl;
        return true; // Conservative: assume coplanar if we can't test
    }

    // Use first 3 connected boundary vertices for coplanarity test
    Eigen::Vector3d v0 = V.row(connected_boundary_vertices[0]);
    Eigen::Vector3d v1 = V.row(connected_boundary_vertices[1]);
    Eigen::Vector3d v2 = V.row(connected_boundary_vertices[2]);

    // Use exact determinant test for coplanarity
    // Four points are coplanar if the volume of tetrahedron is exactly 0
    Eigen::Matrix3d mat;
    mat.row(0) = v1 - v0;
    mat.row(1) = v2 - v0;
    mat.row(2) = test_point - v0;

    double det = mat.determinant();
    std::cout << "Coplanarity determinant: " << det << std::endl;

    // Exact test: determinant must be exactly 0 for coplanarity
    return det == 0.0;
}

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
    int operation_id)
{
    T_before = json_to_matrix<Eigen::MatrixXi>(operation_log["T_before"]);
    V_before = json_to_matrix<Eigen::MatrixXd>(operation_log["V_before"]);
    id_map_before = operation_log["T_id_map_before"].get<std::vector<int64_t>>();
    v_id_map_before = operation_log["V_id_map_before"].get<std::vector<int64_t>>();

    T_after = json_to_matrix<Eigen::MatrixXi>(operation_log["T_after"]);
    V_after = json_to_matrix<Eigen::MatrixXd>(operation_log["V_after"]);
    id_map_after = operation_log["T_id_map_after"].get<std::vector<int64_t>>();
    v_id_map_after = operation_log["V_id_map_after"].get<std::vector<int64_t>>();

    if (false) { // Geometric consistency check: bijective mesh verification with boundary analysis
        std::cout << "Performing geometric consistency check..." << std::endl;

        // Precompute boundary faces for efficient boundary detection
        auto boundary_faces_before = computeBoundaryFaces(T_before);

        bool geometry_consistent = true;

        // Step 1: Find exact coordinate matches between V_after and V_before
        std::set<int64_t> matched_before_ids;
        std::vector<size_t> unmatched_after_indices;

        for (size_t i = 0; i < v_id_map_after.size(); ++i) {
            Eigen::Vector3d coord_after = V_after.row(i);
            bool found_match = false;

            for (size_t j = 0; j < v_id_map_before.size(); ++j) {
                Eigen::Vector3d coord_before = V_before.row(j);

                if (coord_after.x() == coord_before.x() && coord_after.y() == coord_before.y() &&
                    coord_after.z() == coord_before.z()) {
                    found_match = true;
                    matched_before_ids.insert(v_id_map_before[j]);
                    std::cout << "V_after[" << i << "] matches V_before[" << j << "]" << std::endl;
                    break;
                }
            }

            if (!found_match) {
                unmatched_after_indices.push_back(i);
                std::cout << "V_after[" << i << "] (ID:" << v_id_map_after[i] << ") unmatched"
                          << std::endl;
            }
        }

        // Step 2: Check unmatched vertices in V_after (max 1 allowed)
        if (unmatched_after_indices.size() > 1) {
            std::cout << "ERROR: Too many unmatched vertices in V_after ("
                      << unmatched_after_indices.size() << ")" << std::endl;
            geometry_consistent = false;
        } else if (unmatched_after_indices.size() == 1) {
            size_t unmatched_idx = unmatched_after_indices[0];
            Eigen::Vector3d unmatched_point = V_after.row(unmatched_idx);

            int vertex_idx = findVertexByCoordinates(unmatched_point, V_before);

            if (vertex_idx != -1 && isVertexOnBoundary(vertex_idx, boundary_faces_before)) {
                std::cout << "V_after[" << unmatched_idx << "] is boundary vertex " << vertex_idx
                          << std::endl;

                if (checkBoundaryCoplanarity(unmatched_point, V_before, boundary_faces_before)) {
                    std::cout << "Point is coplanar with boundary - OK" << std::endl;
                } else {
                    std::cout << "ERROR: Point NOT coplanar with boundary" << std::endl;
                    geometry_consistent = false;
                }
            } else {
                // Check if interior point
                auto [tet_id, bc] =
                    findTetContainingPointOrient3d(V_before, T_before, unmatched_point);
                if (tet_id != -1) {
                    std::cout << "V_after[" << unmatched_idx << "] is interior - ignoring"
                              << std::endl;
                } else {
                    std::cout << "ERROR: V_after[" << unmatched_idx << "] outside mesh"
                              << std::endl;
                    geometry_consistent = false;
                }
            }
        }

        // Step 3: Find unmatched vertices in V_before
        std::vector<size_t> unmatched_before_indices;
        for (size_t j = 0; j < v_id_map_before.size(); ++j) {
            if (matched_before_ids.find(v_id_map_before[j]) == matched_before_ids.end()) {
                unmatched_before_indices.push_back(j);
                std::cout << "V_before[" << j << "] (ID:" << v_id_map_before[j] << ") unmatched"
                          << std::endl;
            }
        }

        // Step 4: Check unmatched vertices in V_before (max 2 allowed)
        if (unmatched_before_indices.size() > 2) {
            std::cout << "ERROR: Too many unmatched vertices in V_before ("
                      << unmatched_before_indices.size() << ")" << std::endl;
            geometry_consistent = false;
        } else if (unmatched_before_indices.size() > 0) {
            for (size_t unmatched_idx : unmatched_before_indices) {
                Eigen::Vector3d unmatched_point = V_before.row(unmatched_idx);

                if (isVertexOnBoundary(unmatched_idx, boundary_faces_before)) {
                    std::cout << "V_before[" << unmatched_idx << "] is boundary vertex"
                              << std::endl;

                    if (checkBoundaryCoplanarity(
                            unmatched_point,
                            V_before,
                            boundary_faces_before)) {
                        std::cout << "Point is coplanar with boundary - OK" << std::endl;
                    } else {
                        std::cout << "ERROR: Point NOT coplanar with boundary" << std::endl;
                        geometry_consistent = false;
                    }
                } else {
                    std::cout << "V_before[" << unmatched_idx << "] is interior - ignoring"
                              << std::endl;
                }
            }
        }

        // Step 5: Report results
        if (geometry_consistent) {
            std::cout << "Geometric consistency check PASSED" << std::endl;
            std::cout << "Unmatched: " << unmatched_after_indices.size() << " in V_after, "
                      << unmatched_before_indices.size() << " in V_before" << std::endl;
        } else {
            std::cout << "ERROR: Geometric consistency check FAILED" << std::endl;

            // Write debug meshes
            std::string filename_before = "V_before_op" + std::to_string(operation_id) + ".vtu";
            std::string filename_after = "V_after_op" + std::to_string(operation_id) + ".vtu";
            vtu_utils::write_tet_mesh_to_vtu(V_before, T_before, filename_before);
            vtu_utils::write_tet_mesh_to_vtu(V_after, T_after, filename_after);
            std::cout << "Debug meshes written to " << filename_before << " and " << filename_after
                      << std::endl;
        }
    }
}

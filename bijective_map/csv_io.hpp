#pragma once

#include <Eigen/Core>
#include <string>

namespace csv_io {

// Read vertex data from CSV file
Eigen::MatrixXd readVertices(const std::string& filename);

// Read tetrahedron data from CSV file
Eigen::MatrixXi readTetrahedrons(const std::string& filename);

} // namespace csv_io

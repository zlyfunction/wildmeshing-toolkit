#include "csv_io.hpp"
#include <fstream>
#include <sstream>
#include <vector>

namespace csv_io {

Eigen::MatrixXd readVertices(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> data;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream s(line);
        std::string field;
        std::vector<double> row;
        while (std::getline(s, field, ',')) {
            row.push_back(std::stod(field));
        }
        data.push_back(row);
    }

    Eigen::MatrixXd V(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            V(i, j) = data[i][j];
        }
    }
    return V;
}

Eigen::MatrixXi readTetrahedrons(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<int>> data;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream s(line);
        std::string field;
        std::vector<int> row;

        while (std::getline(s, field, ',')) {
            row.push_back(std::stoi(field));
        }
        data.push_back(row);
    }

    Eigen::MatrixXi T(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            T(i, j) = data[i][j];
        }
    }
    return T;
}

} // namespace csv_io

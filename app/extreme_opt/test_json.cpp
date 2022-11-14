#include <iostream>
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

int main()
{
    std::ifstream f("example.json");
    json data = json::parse(f);
    std::cout << data["data"][0];
    double E_avg = 12.0, E_max = 209.3;
    data["data"].push_back({{"E_avg", E_avg}, {"E_max", E_max}});

    data["model_name"] = "know1";
    std::ofstream fout("example_out.json");
    fout << std::setw(4) << data << std::endl;
}
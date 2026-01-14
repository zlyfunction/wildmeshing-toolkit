#include <jse/jse.h>
#include <CLI/CLI.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <wmtk/Mesh.hpp>
#include <wmtk/TetMesh.hpp>
#include <wmtk/utils/Logger.hpp>
#include <wmtk/components/input/input.hpp>
#include <wmtk/components/output/output.hpp>
#include <wmtk/components/tet_remeshing/tet_remeshing.hpp>
#include <wmtk/components/utils/resolve_path.hpp>
#include "tet_remeshing_spec.hpp"
#ifdef WMTK_RECORD_OPERATIONS
#include <wmtk/Record_Operations.hpp>
#endif
// For volume computation
#include <igl/volume.h>
#include <iomanip>
using namespace wmtk;
namespace fs = std::filesystem;
using wmtk::components::utils::resolve_paths;
int main(int argc, char* argv[])
{
    CLI::App app{argv[0]};
    app.ignore_case();
    fs::path json_input_file;
    app.add_option("-j, --json", json_input_file, "json specification file")
        ->required(true)
        ->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv);
    nlohmann::json j;
    {
        std::ifstream ifs(json_input_file);
        j = nlohmann::json::parse(ifs);
        jse::JSE spec_engine;
        bool r = spec_engine.verify_json(j, tet_remeshing_spec);
        if (!r) {
            wmtk::logger().error("{}", spec_engine.log2str());
            return 1;
        } else {
            j = spec_engine.inject_defaults(j, tet_remeshing_spec);
        }
    }
#ifdef WMTK_RECORD_OPERATIONS
    std::string model_name = "default";
    if (j["input"].is_string()) {
        fs::path input_path = j["input"].get<std::string>();
        if (input_path.has_filename()) {
            model_name = input_path.stem().string();
        }
    }
    OperationLogPath = generatePathNameWithModelName(model_name);
    initializeBatchLogging();
#endif
    const fs::path input_file = resolve_paths(json_input_file, {j["input_path"], j["input"]});
    std::shared_ptr<Mesh> mesh_in = wmtk::components::input::input(input_file, true);
    // Check that it's a tet mesh
    if (mesh_in->top_simplex_type() != PrimitiveType::Tetrahedron) {
        wmtk::logger().error(
            "tet_remeshing only works on tetrahedral meshes, but got {}",
            primitive_type_name(mesh_in->top_simplex_type()));
        return 1;
    }
    attribute::MeshAttributeHandle pos_handle =
        mesh_in->get_attribute_handle<double>("vertices", PrimitiveType::Vertex);
    Mesh& mesh = *mesh_in;
    // Compute input mesh statistics
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "INPUT MESH STATISTICS" << std::endl;
        std::cout << "========================================" << std::endl;
        TetMesh& tet_mesh = static_cast<TetMesh&>(mesh);
        auto [T, V] = tet_mesh.get_TV();
        std::cout << "Number of tetrahedra: " << T.rows() << std::endl;
        std::cout << "Number of vertices: " << V.rows() << std::endl;
        Eigen::VectorXd volumes;
        igl::volume(V, T, volumes);
        double min_vol = volumes.array().abs().minCoeff();
        double max_vol = volumes.array().abs().maxCoeff();
        double avg_vol = volumes.array().abs().mean();
        int neg_vol_count = (volumes.array() < 0).count();
        int zero_vol_count = (volumes.array().abs() < 1e-15).count();
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Minimum absolute volume: " << min_vol << std::endl;
        std::cout << "Maximum absolute volume: " << max_vol << std::endl;
        std::cout << "Average absolute volume: " << avg_vol << std::endl;
        std::cout << "Number of negative volume tets: " << neg_vol_count << std::endl;
        std::cout << "Number of near-zero volume tets (|vol| < 1e-15): " << zero_vol_count
                  << std::endl;
        if (neg_vol_count > 0) {
            std::cout << "WARNING: Input mesh contains inverted tetrahedra!" << std::endl;
        }
        if (zero_vol_count > 0) {
            std::cout << "WARNING: Input mesh contains degenerate tetrahedra!" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }
    // tet remeshing
    {
        using namespace components::tet_remeshing;
        TetRemeshingOptions options;
        options.position_handle = pos_handle;
        options.length_rel = j["length_rel"];
        const double env_size = j["envelope_size"];
        if (env_size >= 0) {
            options.envelope_size = j["envelope_size"];
        }
        options.lock_boundary = j["lock_boundary"];
        options.check_inversions = j["check_inversion"];
        options.enable_split = j["enable_split"];
        options.enable_collapse = j["enable_collapse"];
        options.enable_swap = j["enable_swap"];
        options.iterations = j["iterations"];
        tet_remeshing(mesh, options);
    }
    wmtk::components::output::output(mesh, j["output"], pos_handle);
    // Output mesh statistics
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "OUTPUT MESH STATISTICS" << std::endl;
        std::cout << "========================================" << std::endl;
        TetMesh& tet_mesh = static_cast<TetMesh&>(mesh);
        auto [T, V] = tet_mesh.get_TV();
        std::cout << "Number of tetrahedra: " << T.rows() << std::endl;
        std::cout << "Number of vertices: " << V.rows() << std::endl;
        Eigen::VectorXd volumes;
        igl::volume(V, T, volumes);
        double min_vol = volumes.array().abs().minCoeff();
        double max_vol = volumes.array().abs().maxCoeff();
        double avg_vol = volumes.array().abs().mean();
        int neg_vol_count = (volumes.array() < 0).count();
        int zero_vol_count = (volumes.array().abs() < 1e-15).count();
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Minimum absolute volume: " << min_vol << std::endl;
        std::cout << "Maximum absolute volume: " << max_vol << std::endl;
        std::cout << "Average absolute volume: " << avg_vol << std::endl;
        std::cout << "Number of negative volume tets: " << neg_vol_count << std::endl;
        std::cout << "Number of near-zero volume tets (|vol| < 1e-15): " << zero_vol_count
                  << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    const std::string report = j["report"];
    if (!report.empty()) {
        nlohmann::json out_json;
        out_json["stats"]["vertices"] = mesh.get_all(PrimitiveType::Vertex).size();
        out_json["stats"]["edges"] = mesh.get_all(PrimitiveType::Edge).size();
        out_json["stats"]["triangles"] = mesh.get_all(PrimitiveType::Triangle).size();
        out_json["stats"]["tets"] = mesh.get_all(PrimitiveType::Tetrahedron).size();
        out_json["input"] = j;
        std::ofstream ofs(report);
        ofs << std::setw(4) << out_json;
    }
#ifdef WMTK_RECORD_OPERATIONS
    finalizeBatchLogging();
#endif
    return 0;
}

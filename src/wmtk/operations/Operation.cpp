#include "Operation.hpp"

#ifdef WMTK_RECORD_OPERATIONS
#include <wmtk/Record_Operations.hpp>
#endif

#include <nlohmann/json.hpp>
#include <wmtk/Mesh.hpp>
#include <wmtk/multimesh/MultiMeshVisitor.hpp>
#include <wmtk/operations/utils/local_joint_flatten.hpp>
#include <wmtk/simplex/closed_star.hpp>
#include <wmtk/simplex/top_dimension_cofaces.hpp>
#include <wmtk/utils/TupleInspector.hpp>

// it's ugly but for teh visitor we need these included
#include <wmtk/EdgeMesh.hpp>
#include <wmtk/PointMesh.hpp>
#include <wmtk/TetMesh.hpp>
#include <wmtk/TriMesh.hpp>
using json = nlohmann::json;

// for Debugging output
#include <igl/doublearea.h>
#include <igl/writeOBJ.h>

namespace wmtk::operations {


Operation::Operation(Mesh& mesh)
    : m_mesh(mesh)
    , m_invariants(mesh)
{}

Operation::~Operation() = default;


std::shared_ptr<operations::AttributeTransferStrategyBase> Operation::get_transfer_strategy(
    const attribute::MeshAttributeHandle& attribute)
{
    assert(attribute.is_same_mesh(mesh()));

    for (auto& s : m_attr_transfer_strategies) {
        if (s->matches_attribute(attribute)) return s;
    }

    throw std::runtime_error("unable to find attribute");
}

void Operation::set_transfer_strategy(
    const attribute::MeshAttributeHandle& attribute,
    const std::shared_ptr<operations::AttributeTransferStrategyBase>& other)
{
    assert(attribute.is_same_mesh(mesh()));

    for (auto& s : m_attr_transfer_strategies) {
        if (s->matches_attribute(attribute)) {
            s = other;
            return;
        }
    }

    throw std::runtime_error("unable to find attribute");
}

void Operation::add_transfer_strategy(
    const std::shared_ptr<operations::AttributeTransferStrategyBase>& other)
{
    m_attr_transfer_strategies.emplace_back(other);
}

std::vector<simplex::Simplex> Operation::operator()(const simplex::Simplex& simplex)
{
    if (!before(simplex)) {
        return {};
    }

    const auto simplex_resurrect = simplex;

    auto scope = mesh().create_scope();
    assert(simplex.primitive_type() == primitive_type());


    auto unmods = unmodified_primitives(simplex_resurrect);


    auto mods = execute(simplex_resurrect);


    if (!mods.empty()) { // success should be marked here

        if (operation_name != "MeshConsolidate") apply_attribute_transfer(mods);

        if (after(unmods, mods)) {
            // store local atlas for retrieval
#ifdef WMTK_RECORD_OPERATIONS
            if (m_record && operation_name != "MeshConsolidate") {
                // create a local atlas file
                // std::cout << "operation " << operation_name << " is successful\n";
                std::string filename = OperationLogPath + OperationLogPrefix +
                                       std::to_string(succ_operations_count) + ".json";
                std::ofstream operation_log_file(filename);
                json operation_log;

                if (operation_log_file.is_open()) {
                    // save operation_name
                    operation_log["operation_name"] = operation_name;

                    // helper function to convert matrix to json
                    auto matrix_to_json = [](const auto& matrix) {
                        json result = json::array();
                        for (int i = 0; i < matrix.rows(); ++i) {
                            json row = json::array();
                            for (int j = 0; j < matrix.cols(); ++j) {
                                row.push_back(matrix(i, j));
                            }
                            result.push_back(row);
                        }
                        return result;
                    };

                    ////////////////////////////
                    // handle triangle mesh
                    ////////////////////////////
                    if (mesh().top_simplex_type() == PrimitiveType::Triangle) {
                        // get local mesh before and after the operation
                        auto [F_after, V_after, id_map_after, v_id_map_after] =
                            utils::get_local_trimesh(static_cast<const TriMesh&>(mesh()), mods[0]);

                        auto [F_before, V_before, id_map_before, v_id_map_before] =
                            mesh().parent_scope(
                                [&](const simplex::Simplex& s) {
                                    if (operation_name == "EdgeCollapse")
                                        return utils::get_local_trimesh_before_collapse(
                                            static_cast<const TriMesh&>(mesh()),
                                            s);
                                    return utils::get_local_trimesh(
                                        static_cast<const TriMesh&>(mesh()),
                                        s);
                                },
                                simplex);

                        auto to_three_cols = [](const Eigen::MatrixXd& V) {
                            if (V.cols() == 2) {
                                Eigen::MatrixXd V_temp(V.rows(), 3);
                                V_temp << V, Eigen::VectorXd::Zero(V.rows());
                                return V_temp;
                            } else {
                                return V;
                            }
                        };

                        if (operation_name == "EdgeCollapse") {
                            // add different cases for different boundary conditions
                            auto [is_bd_v0, is_bd_v1] = mesh().parent_scope(
                                [&](const simplex::Simplex& s) {
                                    return std::make_tuple(
                                        mesh().is_boundary(
                                            simplex::Simplex::vertex(mesh(), s.tuple())),
                                        mesh().is_boundary(simplex::Simplex::vertex(
                                            mesh(),
                                            mesh().switch_tuple(
                                                s.tuple(),
                                                PrimitiveType::Vertex))));
                                },
                                simplex);

                            Eigen::MatrixXd UV_joint;
                            std::vector<int64_t> v_id_map_joint;

                            if (true) {
                                igl::writeOBJ(
                                    OperationLogPath + "/VF_before_" +
                                        std::to_string(succ_operations_count) + ".obj",
                                    to_three_cols(V_before),
                                    F_before);
                                igl::writeOBJ(
                                    OperationLogPath + "/VF_after_" +
                                        std::to_string(succ_operations_count) + ".obj",
                                    to_three_cols(V_after),
                                    F_after);
                            }
                            utils::local_joint_flatten(
                                F_before,
                                V_before,
                                v_id_map_before,
                                F_after,
                                V_after,
                                v_id_map_after,
                                UV_joint,
                                v_id_map_joint,
                                is_bd_v0,
                                is_bd_v1);
                            if (true) {
                                igl::writeOBJ(
                                    OperationLogPath + "/UV_after_" +
                                        std::to_string(succ_operations_count) + ".obj",
                                    to_three_cols(UV_joint),
                                    F_after);
                                igl::writeOBJ(
                                    OperationLogPath + "/UV_before_" +
                                        std::to_string(succ_operations_count) + ".obj",
                                    to_three_cols(UV_joint),
                                    F_before);
                            }
                            operation_log["UV_joint"]["rows"] = UV_joint.rows();
                            operation_log["UV_joint"]["values"] = matrix_to_json(UV_joint);
                            operation_log["v_id_map_joint"] = v_id_map_joint;
                            operation_log["F_after"]["rows"] = F_after.rows();
                            operation_log["F_after"]["values"] = matrix_to_json(F_after);
                            operation_log["F_before"]["rows"] = F_before.rows();
                            operation_log["F_before"]["values"] = matrix_to_json(F_before);
                            operation_log["F_id_map_after"] = id_map_after;
                            operation_log["F_id_map_before"] = id_map_before;

                            // TODO: add an after check here just to make sure this operation can be
                            // localy parametrized without problem
                            Eigen::VectorXd dbarea_before, dbarea_after;
                            igl::doublearea(UV_joint, F_before, dbarea_before);
                            igl::doublearea(UV_joint, F_after, dbarea_after);

                            if (dbarea_before.minCoeff() < 0) {
                                // std::cerr << "negative area in F_before detected\n";

                                scope.mark_failed();
                                return {};
                            }

                            if (dbarea_after.minCoeff() < 0) {
                                // std::cerr << "negative area in F_after detected\n";

                                scope.mark_failed();
                                return {};
                            }


                        } else { // for other operations

                            if (operation_name == "AttributesUpdate") {
                                Eigen::MatrixXd UV_joint;
                                utils::local_joint_flatten_smoothing(
                                    F_before,
                                    V_before,
                                    V_after,
                                    F_after,
                                    UV_joint);
                                V_before = UV_joint;
                                V_after = UV_joint;

                                v_id_map_before.push_back(v_id_map_before[0]);
                                v_id_map_after.push_back(v_id_map_after[0]);
                            }

                            if (operation_name == "TriEdgeSwap") {
                                Eigen::MatrixXd UV_joint;
                                Eigen::VectorXi local_map;
                                utils::local_joint_flatten_swap(
                                    V_before,
                                    V_after,
                                    F_before,
                                    F_after,
                                    UV_joint,
                                    local_map);

                                V_before = UV_joint;
                                V_after.resize(V_before.rows(), V_before.cols());
                                for (int i = 0; i < 4; ++i) {
                                    V_after.row(i) = V_before.row(local_map(i));
                                }
                            }

                            // log the mesh before and after the operation
                            operation_log["F_after"]["rows"] = F_after.rows();
                            operation_log["F_after"]["values"] = matrix_to_json(F_after);
                            operation_log["V_after"]["rows"] = V_after.rows();
                            operation_log["V_after"]["values"] = matrix_to_json(V_after);
                            operation_log["F_id_map_after"] = id_map_after;
                            operation_log["V_id_map_after"] = v_id_map_after;

                            operation_log["F_before"]["rows"] = F_before.rows();
                            operation_log["F_before"]["values"] = matrix_to_json(F_before);
                            operation_log["V_before"]["rows"] = V_before.rows();
                            operation_log["V_before"]["values"] = matrix_to_json(V_before);
                            operation_log["F_id_map_before"] = id_map_before;
                            operation_log["V_id_map_before"] = v_id_map_before;


                            // TODO: add an after check here just to make sure this operation can be
                            // localy parametrized without problem
                            Eigen::VectorXd dbarea_before, dbarea_after;
                            igl::doublearea(V_before, F_before, dbarea_before);
                            igl::doublearea(V_after, F_after, dbarea_after);

                            if (dbarea_before.minCoeff() < 0) {
                                // std::cerr << "negative area in F_before detected\n";

                                scope.mark_failed();
                                return {};
                            }

                            if (dbarea_after.minCoeff() < 0) {
                                // std::cerr << "negative area in F_after detected\n";

                                scope.mark_failed();
                                return {};
                            }
                        }
                    }

                    // TODO: get a larger json file to do this:
                    // op_logs_js["op_log"].push_back(operation_log);
                    operation_log_file << operation_log.dump(4);
                    operation_log_file.close();
                } else {
                    std::cerr << "unable to open file " << filename << " for writing\n";
                }
                succ_operations_count++;
                // std::cout << "total successful operations: " << succ_operations_count << "\n";

            } // end if (m_record)
#endif
            return mods; // scope destructor is called
        }
    }
    scope.mark_failed();
    return {}; // scope destructor is called
}

bool Operation::before(const simplex::Simplex& simplex) const
{
#if defined(WMTK_ENABLE_HASH_UPDATE)
    const attribute::Accessor<int64_t> accessor = hash_accessor();

    if (!mesh().is_valid(simplex.tuple())) { // TODO: chang to is_removed and resurrect later
        return false;
    }
#else

    if (mesh().is_removed(simplex.tuple()) || !mesh().is_valid(simplex.tuple()) ||
        !mesh().is_valid(simplex)) {
        return false;
    }
#endif

    const auto simplex_resurrect = simplex;

    // map simplex to the invariant mesh
    const Mesh& invariant_mesh = m_invariants.mesh();

    if (&invariant_mesh == &mesh()) {
        if (!m_invariants.before(simplex_resurrect)) {
            return false;
        }
    } else {
        // TODO check if this is correct
        const std::vector<simplex::Simplex> invariant_simplices =
            m_mesh.map(invariant_mesh, simplex_resurrect);

        for (const simplex::Simplex& s : invariant_simplices) {
            if (!m_invariants.before(s)) {
                return false;
            }
        }
    }

    return true;
}

bool Operation::after(
    const std::vector<simplex::Simplex>& unmods,
    const std::vector<simplex::Simplex>& mods) const
{
    return m_invariants.directly_modified_after(unmods, mods);
}

void Operation::apply_attribute_transfer(const std::vector<simplex::Simplex>& direct_mods)
{
    simplex::SimplexCollection all(m_mesh);
    for (const auto& s : direct_mods) {
        if (!s.tuple().is_null()) {
            all.add(simplex::closed_star(m_mesh, s, false));
        }
    }
    all.sort_and_clean();
    for (const auto& at_ptr : m_attr_transfer_strategies) {
        if (&m_mesh == &(at_ptr->mesh())) {
            for (const auto& s : all.simplex_vector()) {
                if (s.primitive_type() == at_ptr->primitive_type()) {
                    at_ptr->run(s);
                }
            }
        } else {
            auto& at_mesh = at_ptr->mesh();
            auto at_mesh_simplices = m_mesh.map(at_mesh, direct_mods);

            simplex::SimplexCollection at_mesh_all(at_mesh);
            for (const auto& s : at_mesh_simplices) {
                at_mesh_all.add(simplex::closed_star(at_mesh, s));
            }

            at_mesh_all.sort_and_clean();

            for (const auto& s : at_mesh_all.simplex_vector()) {
                if (s.primitive_type() == at_ptr->primitive_type()) {
                    at_ptr->run(s);
                }
            }
        }
    }
}



void Operation::reserve_enough_simplices()
{
    // by default assume we'll at most create N * capacity
    constexpr static int64_t default_preallocation_size = 3;

    auto run = [&](auto&& m) {
        if constexpr (!std::is_const_v<std::remove_reference_t<decltype(m)>>) {
            auto cap = m.m_attribute_manager.m_capacities;
            for (auto& v : cap) {
                v *= default_preallocation_size;
            }
            m.reserve_more_attributes(cap);
        }
    };
    multimesh::MultiMeshVisitor visitor(run);
    visitor.execute_from_root(mesh());
}

} // namespace wmtk::operations

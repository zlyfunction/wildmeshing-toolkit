#include "tet_remeshing.hpp"
#include <bitset>
#include <wmtk/Mesh.hpp>
#include <wmtk/Scheduler.hpp>
#include <wmtk/components/utils/get_attributes.hpp>
#include <wmtk/invariants/EdgeValenceInvariant.hpp>
#include <wmtk/invariants/EnvelopeInvariant.hpp>
#include <wmtk/invariants/InteriorSimplexInvariant.hpp>
#include <wmtk/invariants/InvariantCollection.hpp>
#include <wmtk/invariants/MaxEdgeLengthInvariant.hpp>
#include <wmtk/invariants/MultiMeshLinkConditionInvariant.hpp>
#include <wmtk/invariants/MultiMeshMapValidInvariant.hpp>
#include <wmtk/invariants/SelfIntersectionInvariant.hpp>
#include <wmtk/invariants/SimplexInversionInvariant.hpp>
#include <wmtk/invariants/Swap32EnergyBeforeInvariantDouble.hpp>
#include <wmtk/invariants/Swap44EnergyBeforeInvariantDouble.hpp>
#include <wmtk/invariants/Swap56EnergyBeforeInvariantDouble.hpp>
#include <wmtk/invariants/TodoInvariant.hpp>
#include <wmtk/multimesh/MultiMeshVisitor.hpp>
#include <wmtk/multimesh/consolidate.hpp>
#include <wmtk/operations/AttributesUpdate.hpp>
#include <wmtk/operations/EdgeCollapse.hpp>
#include <wmtk/operations/EdgeSplit.hpp>
#include <wmtk/operations/MeshConsolidate.hpp>
#include <wmtk/operations/MinOperationSequence.hpp>
#include <wmtk/operations/OrOperationSequence.hpp>
#include <wmtk/operations/attribute_new/CollapseNewAttributeStrategy.hpp>
#include <wmtk/operations/attribute_new/NewAttributeStrategy.hpp>
#include <wmtk/operations/attribute_new/SplitNewAttributeStrategy.hpp>
#include <wmtk/operations/attribute_update/AttributeTransferStrategy.hpp>
#include <wmtk/operations/composite/TetEdgeSwap.hpp>
#include <wmtk/utils/Logger.hpp>
namespace wmtk::components::tet_remeshing {
void tet_remeshing(Mesh& mesh_in, const TetRemeshingOptions& options)
{
    if (mesh_in.top_simplex_type() != PrimitiveType::Tetrahedron) {
        log_and_throw_error(
            "tet remeshing works only for tet meshes: {}",
            primitive_type_name(mesh_in.top_simplex_type()));
    }
    if (!mesh_in.is_multi_mesh_root()) {
        log_and_throw_error("The mesh passed in tet_remeshing must be the root mesh");
    }
    attribute::MeshAttributeHandle position_handle = options.position_handle;
    std::vector<attribute::MeshAttributeHandle> other_position_handles =
        options.other_position_handles;
    Mesh& mesh = position_handle.mesh();
    std::vector<attribute::MeshAttributeHandle> inversion_position_handles;
    if (options.check_inversions) {
        if (position_handle.mesh().top_cell_dimension() == position_handle.dimension()) {
            logger().info("Adding inversion check on remeshing mesh.");
            inversion_position_handles.emplace_back(position_handle);
        }
        for (auto& h : other_position_handles) {
            if (h.mesh().top_cell_dimension() == h.dimension()) {
                logger().info("Adding inversion check on other mesh.");
                inversion_position_handles.emplace_back(h);
            }
        }
        if (inversion_position_handles.empty()) {
            logger().warn("Tet remeshing should check for inversions but there was no "
                          "position handle that is valid for inversion checks.");
        }
    }
    std::vector<attribute::MeshAttributeHandle> pass_through_attributes =
        options.pass_through_attributes;
    for (auto& h : other_position_handles) {
        pass_through_attributes.emplace_back(h);
    }
    /////////////////////////////////////////////
    auto visited_edge_flag =
        mesh.register_attribute<char>("visited_edge", PrimitiveType::Edge, 1, false, char(1));
    auto update_flag_func = [](Eigen::Ref<const Eigen::MatrixXd> P) -> Eigen::VectorX<char> {
        assert(P.cols() == 2);
        assert(P.rows() == 3);
        return Eigen::VectorX<char>::Constant(1, char(1));
    };
    auto tag_update =
        std::make_shared<wmtk::operations::SingleAttributeTransferStrategy<char, double>>(
            visited_edge_flag,
            position_handle,
            update_flag_func);
    //////////////////////////////////
    // Storing edge lengths
    auto edge_length_attribute =
        mesh.register_attribute<double>("edge_length", PrimitiveType::Edge, 1);
    auto edge_length_accessor = mesh.create_accessor(edge_length_attribute.as<double>());
    // Edge length update
    auto compute_edge_length = [](Eigen::Ref<const Eigen::MatrixXd> P) -> Eigen::VectorXd {
        assert(P.cols() == 2);
        assert(P.rows() == 3);
        return Eigen::VectorXd::Constant(1, (P.col(0) - P.col(1)).norm());
    };
    auto edge_length_update =
        std::make_shared<wmtk::operations::SingleAttributeTransferStrategy<double, double>>(
            edge_length_attribute,
            position_handle,
            compute_edge_length);
    edge_length_update->run_on_all();
    //////////////////////////////////
    // computing bbox diagonal
    Eigen::VectorXd bmin(position_handle.dimension());
    bmin.setConstant(std::numeric_limits<double>::max());
    Eigen::VectorXd bmax(position_handle.dimension());
    bmax.setConstant(std::numeric_limits<double>::lowest());
    auto pt_accessor = mesh.create_const_accessor<double>(position_handle);
    const auto vertices = mesh.get_all(PrimitiveType::Vertex);
    for (const auto& v : vertices) {
        const auto p = pt_accessor.vector_attribute(v);
        for (int64_t d = 0; d < bmax.size(); ++d) {
            bmin[d] = std::min(bmin[d], p[d]);
            bmax[d] = std::max(bmax[d], p[d]);
        }
    }
    const double bbdiag = (bmax - bmin).norm();
    const double length_abs = bbdiag * options.length_rel;
    wmtk::logger().info(
        "bbox max {}, bbox min {}, diag {}, target edge length {}",
        bmax,
        bmin,
        bbdiag,
        length_abs);
    pass_through_attributes.push_back(edge_length_attribute);
    //////////////////////////invariants
    auto invariant_link_condition = std::make_shared<MultiMeshLinkConditionInvariant>(mesh);
    auto invariant_interior_edge = std::make_shared<invariants::InvariantCollection>(mesh);
    auto invariant_interior_vertex = std::make_shared<invariants::InvariantCollection>(mesh);
    auto set_all_invariants = [&](auto&& m) {
        invariant_interior_edge->add(
            std::make_shared<invariants::InteriorSimplexInvariant>(m, PrimitiveType::Edge));
        invariant_interior_vertex->add(
            std::make_shared<invariants::InteriorSimplexInvariant>(m, PrimitiveType::Vertex));
    };
    wmtk::multimesh::MultiMeshVisitor visitor(set_all_invariants);
    visitor.execute_from_root(mesh);
    auto invariant_mm_map = std::make_shared<MultiMeshMapValidInvariant>(mesh);
    ////////////// positions
    std::vector<attribute::MeshAttributeHandle> position_handles;
    position_handles.emplace_back(position_handle);
    // Prefer boundary vertex if only one endpoint is boundary; otherwise keep the first endpoint
    auto position_collapse_no_mean =
        [](const Eigen::VectorXd& a, const Eigen::VectorXd& b, const std::bitset<2>& bs) {
            if (bs[0] != bs[1]) {
                return bs[0] ? a : b;
            }
            return b;
        };
    auto propagate_position = [](const Eigen::MatrixXd& P) -> Eigen::VectorXd { return P; };
    //////////////////////////////////////////
    // Run iterations of split and collapse
    for (int iter = 0; iter < options.iterations; ++iter) {
        logger().info("Tet remeshing iteration {}/{}", iter + 1, options.iterations);
        // Reset visited flags
        {
            auto accessor = mesh.create_accessor(visited_edge_flag.as<char>());
            for (const auto& e : mesh.get_all(PrimitiveType::Edge)) {
                accessor.scalar_attribute(e) = char(1);
            }
        }
        edge_length_update->run_on_all();
        //////////////////////////////////////////
        // Split long edges
        if (options.enable_split) {
            auto long_edges_first_priority = [&](const simplex::Simplex& s) {
                assert(s.primitive_type() == PrimitiveType::Edge);
                return -edge_length_accessor.const_scalar_attribute(s.tuple());
            };
            auto todo_split = std::make_shared<TodoLargerInvariant>(
                mesh,
                edge_length_attribute.as<double>(),
                length_abs * 4.0 / 3.0);
            auto split = std::make_shared<wmtk::operations::EdgeSplit>(mesh);
            split->add_invariant(todo_split);
            split->set_new_attribute_strategy(
                visited_edge_flag,
                wmtk::operations::SplitBasicStrategy::None,
                wmtk::operations::SplitRibBasicStrategy::None);
            split->add_transfer_strategy(tag_update);
            split->set_priority(long_edges_first_priority);
            split->add_transfer_strategy(edge_length_update);
            for (const auto& pos_handle : position_handles) {
                split->set_new_attribute_strategy(pos_handle);
            }
            for (const auto& attr : pass_through_attributes) {
                split->set_new_attribute_strategy(attr);
            }
            for (auto& h : other_position_handles) {
                auto transfer_position =
                    std::make_shared<operations::SingleAttributeTransferStrategy<double, double>>(
                        h,
                        position_handle,
                        propagate_position);
                split->add_transfer_strategy(transfer_position);
            }
            Scheduler scheduler_split;
            SchedulerStats split_stats =
                scheduler_split.run_operation_on_all(*split, visited_edge_flag.as<char>());
            logger().info(
                "Split: Executed {} ops (S/F) {}/{}",
                split_stats.number_of_performed_operations(),
                split_stats.number_of_successful_operations(),
                split_stats.number_of_failed_operations());
        }
        // Reset visited flags after split
        {
            auto accessor = mesh.create_accessor(visited_edge_flag.as<char>());
            for (const auto& e : mesh.get_all(PrimitiveType::Edge)) {
                accessor.scalar_attribute(e) = char(1);
            }
        }
        edge_length_update->run_on_all();
        //////////////////////////////////////////
        // Collapse short edges
        if (options.enable_collapse) {
            auto short_edges_first_priority = [&](const simplex::Simplex& s) {
                assert(s.primitive_type() == PrimitiveType::Edge);
                return edge_length_accessor.const_scalar_attribute(s.tuple());
            };
            auto todo_collapse = std::make_shared<TodoSmallerInvariant>(
                mesh,
                edge_length_attribute.as<double>(),
                length_abs * 4.0 / 5.0);
            auto collapse = std::make_shared<wmtk::operations::EdgeCollapse>(mesh);
            collapse->add_invariant(todo_collapse);
            collapse->add_invariant(invariant_link_condition);
            collapse->add_invariant(invariant_mm_map);
            if (options.envelope_size) {
                const double env_size = bbdiag * options.envelope_size.value();
                bool envelope_added = false;
                if (position_handle.mesh().top_cell_dimension() < position_handle.dimension()) {
                    logger().info("Adding envelope check on collapsing mesh.");
                    collapse->add_invariant(std::make_shared<wmtk::invariants::EnvelopeInvariant>(
                        position_handle,
                        env_size,
                        position_handle));
                    envelope_added = true;
                }
                for (auto& h : other_position_handles) {
                    if (h.mesh().top_cell_dimension() < h.dimension()) {
                        logger().info("Adding envelope check on other mesh.");
                        collapse->add_invariant(
                            std::make_shared<wmtk::invariants::EnvelopeInvariant>(h, env_size, h));
                        envelope_added = true;
                    }
                }
                if (!envelope_added) {
                    logger().warn("Tet remeshing should check for envelope but there was no "
                                  "position handle that is valid for envelope checks.");
                }
            }
            for (auto& h : inversion_position_handles) {
                collapse->add_invariant(
                    std::make_shared<SimplexInversionInvariant<double>>(h.mesh(), h.as<double>()));
            }
            for (auto& h : inversion_position_handles) {
                collapse->add_invariant(
                    std::make_shared<SelfIntersectionInvariant<double>>(h.mesh(), h.as<double>()));
            }
            collapse->set_new_attribute_strategy(
                visited_edge_flag,
                wmtk::operations::CollapseBasicStrategy::None);
            collapse->add_transfer_strategy(tag_update);
            collapse->set_priority(short_edges_first_priority);
            collapse->add_transfer_strategy(edge_length_update);
            if (options.lock_boundary) {
                collapse->add_invariant(invariant_interior_edge);
                for (const auto& pos_handle : position_handles) {
                    auto pos_collapse_strategy =
                        std::make_shared<wmtk::operations::CollapseNewAttributeStrategy<double>>(
                            pos_handle);
                    pos_collapse_strategy->set_strategy(position_collapse_no_mean);
                    pos_collapse_strategy->set_simplex_predicate(
                        wmtk::operations::BasicSimplexPredicate::IsInterior);
                    collapse->set_new_attribute_strategy(pos_handle, pos_collapse_strategy);
                }
            } else {
                for (const auto& pos_handle : position_handles) {
                    auto pos_collapse_strategy =
                        std::make_shared<wmtk::operations::CollapseNewAttributeStrategy<double>>(
                            pos_handle);
                    pos_collapse_strategy->set_strategy(position_collapse_no_mean);
                    pos_collapse_strategy->set_simplex_predicate(
                        wmtk::operations::BasicSimplexPredicate::IsInterior);
                    collapse->set_new_attribute_strategy(pos_handle, pos_collapse_strategy);
                }
            }
            for (const auto& attr : pass_through_attributes) {
                collapse->set_new_attribute_strategy(attr);
            }
            for (auto& h : other_position_handles) {
                auto transfer_position =
                    std::make_shared<operations::SingleAttributeTransferStrategy<double, double>>(
                        h,
                        position_handle,
                        propagate_position);
                collapse->add_transfer_strategy(transfer_position);
            }
            Scheduler scheduler_collapse;
            SchedulerStats collapse_stats =
                scheduler_collapse.run_operation_on_all(*collapse, visited_edge_flag.as<char>());
            logger().info(
                "Collapse: Executed {} ops (S/F) {}/{}",
                collapse_stats.number_of_performed_operations(),
                collapse_stats.number_of_successful_operations(),
                collapse_stats.number_of_failed_operations());
        }
        // Reset visited flags after collapse
        {
            auto accessor = mesh.create_accessor(visited_edge_flag.as<char>());
            for (const auto& e : mesh.get_all(PrimitiveType::Edge)) {
                accessor.scalar_attribute(e) = char(1);
            }
        }
        edge_length_update->run_on_all();
        //////////////////////////////////////////
        // Swap edges
        if (options.enable_swap) {
            auto long_edges_first_priority = [&](const simplex::Simplex& s) {
                assert(s.primitive_type() == PrimitiveType::Edge);
                return -edge_length_accessor.const_scalar_attribute(s.tuple());
            };
            // Create inversion invariant for swap
            auto inversion_invariant = std::make_shared<SimplexInversionInvariant<double>>(
                mesh,
                position_handle.as<double>());
            // Swap 5-6 (edge valence 5)
            auto swap56 = std::make_shared<wmtk::operations::MinOperationSequence>(mesh);
            // Set value function for MinOperationSequence (required, tries operations in order)
            swap56->set_value_function([](int64_t, const simplex::Simplex&) { return 0.0; });
            for (int i = 0; i < 5; ++i) {
                auto swap = std::make_shared<wmtk::operations::composite::TetEdgeSwap>(mesh, i);
                swap->collapse().add_invariant(invariant_link_condition);
                swap->collapse().set_new_attribute_strategy(
                    position_handle,
                    wmtk::operations::CollapseBasicStrategy::CopyOther);
                swap->split().set_new_attribute_strategy(position_handle);
                swap->split().set_new_attribute_strategy(
                    visited_edge_flag,
                    wmtk::operations::SplitBasicStrategy::None,
                    wmtk::operations::SplitRibBasicStrategy::None);
                swap->collapse().set_new_attribute_strategy(
                    visited_edge_flag,
                    wmtk::operations::CollapseBasicStrategy::None);
                swap->add_invariant(std::make_shared<wmtk::Swap56EnergyBeforeInvariantDouble>(
                    mesh,
                    position_handle.as<double>(),
                    i));
                swap->add_transfer_strategy(edge_length_update);
                swap->collapse().add_invariant(inversion_invariant);
                for (const auto& attr : pass_through_attributes) {
                    swap->split().set_new_attribute_strategy(
                        attr,
                        wmtk::operations::SplitBasicStrategy::None,
                        wmtk::operations::SplitRibBasicStrategy::None);
                    swap->collapse().set_new_attribute_strategy(
                        attr,
                        wmtk::operations::CollapseBasicStrategy::None);
                }
                swap56->add_operation(swap);
            }
            swap56->add_invariant(
                std::make_shared<wmtk::invariants::EdgeValenceInvariant>(mesh, 5));
            // Swap 4-4 (edge valence 4)
            auto swap44 = std::make_shared<wmtk::operations::MinOperationSequence>(mesh);
            // Set value function for MinOperationSequence (required, tries operations in order)
            swap44->set_value_function([](int64_t, const simplex::Simplex&) { return 0.0; });
            for (int i = 0; i < 2; ++i) {
                auto swap = std::make_shared<wmtk::operations::composite::TetEdgeSwap>(mesh, i);
                swap->collapse().add_invariant(invariant_link_condition);
                swap->collapse().set_new_attribute_strategy(
                    position_handle,
                    wmtk::operations::CollapseBasicStrategy::CopyOther);
                swap->split().set_new_attribute_strategy(position_handle);
                swap->split().set_new_attribute_strategy(
                    visited_edge_flag,
                    wmtk::operations::SplitBasicStrategy::None,
                    wmtk::operations::SplitRibBasicStrategy::None);
                swap->collapse().set_new_attribute_strategy(
                    visited_edge_flag,
                    wmtk::operations::CollapseBasicStrategy::None);
                swap->add_invariant(std::make_shared<wmtk::Swap44EnergyBeforeInvariantDouble>(
                    mesh,
                    position_handle.as<double>(),
                    i));
                swap->add_transfer_strategy(edge_length_update);
                swap->collapse().add_invariant(inversion_invariant);
                for (const auto& attr : pass_through_attributes) {
                    swap->split().set_new_attribute_strategy(
                        attr,
                        wmtk::operations::SplitBasicStrategy::None,
                        wmtk::operations::SplitRibBasicStrategy::None);
                    swap->collapse().set_new_attribute_strategy(
                        attr,
                        wmtk::operations::CollapseBasicStrategy::None);
                }
                swap44->add_operation(swap);
            }
            swap44->add_invariant(
                std::make_shared<wmtk::invariants::EdgeValenceInvariant>(mesh, 4));
            // Swap 3-2 (edge valence 3)
            auto swap32 = std::make_shared<wmtk::operations::composite::TetEdgeSwap>(mesh, 0);
            swap32->add_invariant(
                std::make_shared<wmtk::invariants::EdgeValenceInvariant>(mesh, 3));
            swap32->add_invariant(std::make_shared<wmtk::Swap32EnergyBeforeInvariantDouble>(
                mesh,
                position_handle.as<double>()));
            swap32->collapse().add_invariant(invariant_link_condition);
            swap32->collapse().set_new_attribute_strategy(
                position_handle,
                wmtk::operations::CollapseBasicStrategy::CopyOther);
            swap32->split().set_new_attribute_strategy(position_handle);
            swap32->split().set_new_attribute_strategy(
                visited_edge_flag,
                wmtk::operations::SplitBasicStrategy::None,
                wmtk::operations::SplitRibBasicStrategy::None);
            swap32->collapse().set_new_attribute_strategy(
                visited_edge_flag,
                wmtk::operations::CollapseBasicStrategy::None);
            swap32->add_transfer_strategy(edge_length_update);
            swap32->collapse().add_invariant(inversion_invariant);
            for (const auto& attr : pass_through_attributes) {
                swap32->split().set_new_attribute_strategy(
                    attr,
                    wmtk::operations::SplitBasicStrategy::None,
                    wmtk::operations::SplitRibBasicStrategy::None);
                swap32->collapse().set_new_attribute_strategy(
                    attr,
                    wmtk::operations::CollapseBasicStrategy::None);
            }
            // Combine all swaps
            auto swap_all = std::make_shared<wmtk::operations::OrOperationSequence>(mesh);
            swap_all->add_operation(swap32);
            swap_all->add_operation(swap44);
            swap_all->add_operation(swap56);
            swap_all->add_transfer_strategy(tag_update);
            swap_all->add_invariant(invariant_interior_edge);
            swap_all->set_priority(long_edges_first_priority);
            Scheduler scheduler_swap;
            SchedulerStats swap_stats =
                scheduler_swap.run_operation_on_all(*swap_all, visited_edge_flag.as<char>());
            logger().info(
                "Swap: Executed {} ops (S/F) {}/{}",
                swap_stats.number_of_performed_operations(),
                swap_stats.number_of_successful_operations(),
                swap_stats.number_of_failed_operations());
        }
    }
    //////////////////////////////////////////
    auto op_consolidate = wmtk::operations::MeshConsolidate(mesh);
    op_consolidate(simplex::Simplex(mesh, PrimitiveType::Vertex, Tuple()));
    logger().info("Tet remeshing completed.");
}
void tet_remeshing(
    Mesh& mesh,
    const attribute::MeshAttributeHandle& position_handle,
    const double length_rel,
    std::optional<bool> lock_boundary,
    std::optional<double> envelope_size,
    bool check_inversion,
    bool enable_split,
    bool enable_collapse,
    bool enable_swap,
    int iterations,
    const std::vector<attribute::MeshAttributeHandle>& pass_through)
{
    TetRemeshingOptions options;
    options.position_handle = position_handle;
    options.length_rel = length_rel;
    if (lock_boundary) {
        options.lock_boundary = lock_boundary.value();
    }
    options.envelope_size = envelope_size;
    options.check_inversions = check_inversion;
    options.enable_split = enable_split;
    options.enable_collapse = enable_collapse;
    options.enable_swap = enable_swap;
    options.iterations = iterations;
    options.pass_through_attributes = pass_through;
    tet_remeshing(mesh, options);
}
} // namespace wmtk::components::tet_remeshing

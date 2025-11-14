#include "isotropic_remeshing.hpp"

#include <wmtk/EdgeMesh.hpp>
#include <wmtk/Scheduler.hpp>
#include <wmtk/TriMesh.hpp>
#include <wmtk/invariants/FusionEdgeInvariant.hpp>
#include <wmtk/invariants/InteriorSimplexInvariant.hpp>
#include <wmtk/invariants/InvariantCollection.hpp>
#include <wmtk/invariants/MaxEdgeLengthInvariant.hpp>
#include <wmtk/invariants/MinEdgeLengthInvariant.hpp>
#include <wmtk/invariants/MultiMeshLinkConditionInvariant.hpp>
#include <wmtk/invariants/MultiMeshMapValidInvariant.hpp>
#include <wmtk/invariants/SimplexInversionInvariant.hpp>
#include <wmtk/invariants/ValenceImprovementInvariant.hpp>
#include <wmtk/invariants/uvEdgeInvariant.hpp>
#include <wmtk/io/ParaviewWriter.hpp>
#include <wmtk/multimesh/MultiMeshVisitor.hpp>
#include <wmtk/multimesh/consolidate.hpp>
#include <wmtk/operations/AttributesUpdate.hpp>
#include <wmtk/operations/EdgeCollapse.hpp>
#include <wmtk/operations/EdgeSplit.hpp>
#include <wmtk/operations/MeshConsolidate.hpp>
#include <wmtk/operations/attribute_new/CollapseNewAttributeStrategy.hpp>
#include <wmtk/operations/attribute_new/SplitNewAttributeStrategy.hpp>
#include <wmtk/operations/attribute_update/AttributeTransferStrategy.hpp>
#include <wmtk/operations/composite/TriEdgeSwap.hpp>
#include <wmtk/operations/utils/VertexLaplacianSmooth.hpp>
#include <wmtk/operations/utils/VertexTangentialLaplacianSmooth.hpp>
#include <wmtk/utils/Logger.hpp>

#include <wmtk/simplex/RawSimplex.hpp>
#include <wmtk/simplex/Simplex.hpp>
#include <wmtk/simplex/faces_single_dimension.hpp>
#include <wmtk/simplex/link.hpp>
#include <wmtk/simplex/top_dimension_cofaces.hpp>

#include <Eigen/Geometry>
#include <array>
#include <map>
#include <optional>
#include <wmtk/invariants/InvariantCollection.hpp>
#include "IsotropicRemeshingOptions.hpp"

namespace wmtk::components::isotropic_remeshing {
// compute the length relative to the bounding box diagonal
double relative_to_absolute_length(
    const attribute::MeshAttributeHandle& position,
    const double length_rel)
{
    auto pos = position.mesh().create_const_accessor<double>(position);
    const auto vertices = position.mesh().get_all(PrimitiveType::Vertex);
    Eigen::AlignedBox<double, Eigen::Dynamic> bbox(pos.dimension());


    for (const auto& v : vertices) {
        bbox.extend(pos.const_vector_attribute(v));
    }

    const double diag_length = bbox.sizes().norm();

    return length_rel * diag_length;
}

namespace {
SchedulerStats run_reference_collapse(
    operations::EdgeCollapse& op,
    const attribute::MeshAttributeHandle& position,
    const double length_min,
    const double length_max,
    const bool lock_boundary)
{
    SchedulerStats stats;

    auto& mesh = static_cast<TriMesh&>(op.mesh());
    auto pos = position.mesh().create_const_accessor<double>(position);

    const double short_sq = length_min * length_min;
    const double long_sq = length_max * length_max;

    auto gather_one_ring = [&](const Tuple& vertex) {
        std::vector<Tuple> neighbors;
        if (!mesh.is_valid(vertex)) {
            return neighbors;
        }
        const auto ring = simplex::link(mesh, simplex::Simplex::vertex(mesh, vertex))
                              .simplex_vector(PrimitiveType::Vertex);
        neighbors.reserve(ring.size());
        for (const auto& neighbor : ring) {
            neighbors.emplace_back(neighbor.tuple());
        }
        return neighbors;
    };

    auto creates_long_edge = [&](const Tuple& removed, const Tuple& kept) {
        const auto kept_pos = pos.const_vector_attribute(kept);
        const simplex::RawSimplex kept_key(mesh, simplex::Simplex::vertex(mesh, kept));
        const auto neighbors = gather_one_ring(removed);
        for (const Tuple& nbr : neighbors) {
            if (!mesh.is_valid(nbr)) {
                continue;
            }
            const simplex::RawSimplex nbr_key(mesh, simplex::Simplex::vertex(mesh, nbr));
            if (nbr_key == kept_key) {
                continue;
            }
            const auto pn = pos.const_vector_attribute(nbr);
            if ((pn - kept_pos).squaredNorm() > long_sq) {
                return true;
            }
        }
        return false;
    };

    auto vertex_valence = [&](const Tuple& vertex) -> int64_t {
        return static_cast<int64_t>(gather_one_ring(vertex).size());
    };

    bool changed = true;
    int sweep = 0;
    while (changed && sweep < 10) {
        changed = false;
        ++sweep;

        const auto edges = mesh.get_all(PrimitiveType::Edge);
        for (const Tuple& edge : edges) {
            if (!mesh.is_valid(edge)) {
                continue;
            }

            const Tuple v0 = edge;
            const Tuple v1 = mesh.switch_tuple(edge, PrimitiveType::Vertex);
            if (!mesh.is_valid(v0) || !mesh.is_valid(v1)) {
                continue;
            }

            const auto p0 = pos.const_vector_attribute(v0);
            const auto p1 = pos.const_vector_attribute(v1);
            if ((p0 - p1).squaredNorm() >= short_sq) {
                continue;
            }

            const bool b0 = mesh.is_boundary(PrimitiveType::Vertex, v0);
            const bool b1 = mesh.is_boundary(PrimitiveType::Vertex, v1);
            const bool boundary_edge = mesh.is_boundary(PrimitiveType::Edge, edge);

            if (lock_boundary && (b0 || b1 || boundary_edge)) {
                continue;
            }

            if (b0 && b1 && !boundary_edge) {
                continue;
            }

            bool collapse_v0_to_v1 = true;
            bool collapse_v1_to_v0 = true;

            if (b0 && !b1) {
                collapse_v0_to_v1 = false;
            } else if (b1 && !b0) {
                collapse_v1_to_v0 = false;
            }

            if (collapse_v0_to_v1 && creates_long_edge(v0, v1)) {
                collapse_v0_to_v1 = false;
            }
            if (collapse_v1_to_v0 && creates_long_edge(v1, v0)) {
                collapse_v1_to_v0 = false;
            }

            if (!collapse_v0_to_v1 && !collapse_v1_to_v0) {
                continue;
            }

            if (collapse_v0_to_v1 && collapse_v1_to_v0) {
                if (vertex_valence(v0) < vertex_valence(v1)) {
                    collapse_v0_to_v1 = false;
                } else {
                    collapse_v1_to_v0 = false;
                }
            }

            Tuple collapse_tuple;
            if (collapse_v0_to_v1) {
                collapse_tuple = v0;
            } else if (collapse_v1_to_v0) {
                collapse_tuple = v1;
            } else {
                continue;
            }

            auto mods = op(simplex::Simplex(mesh, PrimitiveType::Edge, collapse_tuple));
            if (mods.empty()) {
                stats.fail();
                continue;
            }

            stats.succeed();
            changed = true;
        }
    }

    return stats;
}

SchedulerStats run_reference_split(
    operations::EdgeSplit& op,
    const attribute::MeshAttributeHandle& position,
    const double length_max,
    const bool lock_boundary)
{
    SchedulerStats stats;
    auto& mesh = static_cast<TriMesh&>(op.mesh());
    auto pos = position.mesh().create_const_accessor<double>(position);
    const double long_sq = length_max * length_max;

    bool changed = true;
    int sweep = 0;
    while (changed && sweep < 10) {
        changed = false;
        ++sweep;

        const auto edges = mesh.get_all(PrimitiveType::Edge);
        for (const Tuple& edge : edges) {
            if (!mesh.is_valid(edge)) {
                continue;
            }

            const Tuple v0 = edge;
            const Tuple v1 = mesh.switch_tuple(edge, PrimitiveType::Vertex);
            if (!mesh.is_valid(v0) || !mesh.is_valid(v1)) {
                continue;
            }

            const bool boundary_edge = mesh.is_boundary(PrimitiveType::Edge, edge);
            const bool b0 = mesh.is_boundary(PrimitiveType::Vertex, v0);
            const bool b1 = mesh.is_boundary(PrimitiveType::Vertex, v1);
            if (lock_boundary && (boundary_edge || b0 || b1)) {
                continue;
            }

            const auto p0 = pos.const_vector_attribute(v0);
            const auto p1 = pos.const_vector_attribute(v1);
            if ((p0 - p1).squaredNorm() <= long_sq) {
                continue;
            }

            auto mods = op(simplex::Simplex(mesh, PrimitiveType::Edge, edge));
            if (mods.empty()) {
                stats.fail();
                continue;
            }

            stats.succeed();
            changed = true;
        }
    }

    return stats;
}

SchedulerStats run_reference_swap(operations::composite::TriEdgeSwap& op, const bool lock_boundary)
{
    SchedulerStats stats;
    auto& mesh = static_cast<TriMesh&>(op.mesh());

    std::map<simplex::RawSimplex, int64_t> valence_cache;

    auto ensure_valence = [&](const Tuple& vertex, const simplex::RawSimplex& key) -> int64_t {
        auto [it, inserted] = valence_cache.try_emplace(key, 0);
        if (inserted) {
            it->second =
                static_cast<int64_t>(simplex::link(mesh, simplex::Simplex::vertex(mesh, vertex))
                                         .simplex_vector(PrimitiveType::Vertex)
                                         .size());
        }
        return it->second;
    };

    auto opposite_vertex = [&](const Tuple& face,
                               const simplex::RawSimplex& key0,
                               const simplex::RawSimplex& key1) -> std::optional<Tuple> {
        const auto vertices = simplex::faces_single_dimension_tuples(
            mesh,
            simplex::Simplex(mesh, PrimitiveType::Triangle, face),
            PrimitiveType::Vertex);
        for (const Tuple& v : vertices) {
            const simplex::RawSimplex candidate_key(mesh, simplex::Simplex::vertex(mesh, v));
            if (candidate_key == key0 || candidate_key == key1) {
                continue;
            }
            return v;
        }
        return std::nullopt;
    };

    auto valence_energy = [](int64_t val, int opt) {
        const int64_t diff = val - opt;
        return diff * diff;
    };

    auto optimal_valence = [&](const Tuple& vertex) {
        return mesh.is_boundary(PrimitiveType::Vertex, vertex) ? 4 : 6;
    };

    bool changed = true;
    int sweep = 0;
    while (changed && sweep < 10) {
        changed = false;
        ++sweep;

        const auto edges = mesh.get_all(PrimitiveType::Edge);
        for (const Tuple& edge : edges) {
            if (!mesh.is_valid(edge)) {
                continue;
            }
            if (mesh.is_boundary(PrimitiveType::Edge, edge)) {
                continue;
            }

            const Tuple v0 = edge;
            const Tuple v1 = mesh.switch_tuple(edge, PrimitiveType::Vertex);
            if (!mesh.is_valid(v0) || !mesh.is_valid(v1)) {
                continue;
            }

            if (lock_boundary && (mesh.is_boundary(PrimitiveType::Vertex, v0) ||
                                  mesh.is_boundary(PrimitiveType::Vertex, v1))) {
                continue;
            }

            const simplex::Simplex edge_simplex(mesh, PrimitiveType::Edge, edge);
            const auto faces = simplex::top_dimension_cofaces_tuples(mesh, edge_simplex);
            if (faces.size() != 2) {
                continue;
            }

            const simplex::RawSimplex key0(mesh, simplex::Simplex::vertex(mesh, v0));
            const simplex::RawSimplex key1(mesh, simplex::Simplex::vertex(mesh, v1));

            auto opt_v2 = opposite_vertex(faces[0], key0, key1);
            auto opt_v3 = opposite_vertex(faces[1], key0, key1);
            if (!opt_v2.has_value() || !opt_v3.has_value()) {
                continue;
            }
            const Tuple v2 = opt_v2.value();
            const Tuple v3 = opt_v3.value();

            if (lock_boundary && (mesh.is_boundary(PrimitiveType::Vertex, v2) ||
                                  mesh.is_boundary(PrimitiveType::Vertex, v3))) {
                continue;
            }

            const simplex::RawSimplex key2(mesh, simplex::Simplex::vertex(mesh, v2));
            const simplex::RawSimplex key3(mesh, simplex::Simplex::vertex(mesh, v3));

            const int64_t val0 = ensure_valence(v0, key0);
            const int64_t val1 = ensure_valence(v1, key1);
            const int64_t val2 = ensure_valence(v2, key2);
            const int64_t val3 = ensure_valence(v3, key3);

            const int opt0 = optimal_valence(v0);
            const int opt1 = optimal_valence(v1);
            const int opt2 = optimal_valence(v2);
            const int opt3 = optimal_valence(v3);

            const int64_t energy_before = valence_energy(val0, opt0) + valence_energy(val1, opt1) +
                                          valence_energy(val2, opt2) + valence_energy(val3, opt3);
            const int64_t energy_after =
                valence_energy(val0 - 1, opt0) + valence_energy(val1 - 1, opt1) +
                valence_energy(val2 + 1, opt2) + valence_energy(val3 + 1, opt3);

            if (energy_before <= energy_after) {
                continue;
            }

            auto mods = op(simplex::Simplex(mesh, PrimitiveType::Edge, edge));
            if (mods.empty()) {
                stats.fail();
                continue;
            }

            stats.succeed();
            changed = true;

            valence_cache[key0] = val0 - 1;
            valence_cache[key1] = val1 - 1;
            valence_cache[key2] = val2 + 1;
            valence_cache[key3] = val3 + 1;
        }
    }

    return stats;
}
} // namespace


void isotropic_remeshing(const IsotropicRemeshingOptions& options)
{
    using namespace internal;


    auto position = options.position_attribute;

    if (position.mesh().top_simplex_type() != PrimitiveType::Triangle) {
        log_and_throw_error(
            "isotropic remeshing works only for triangle meshes: {}",
            primitive_type_name(position.mesh().top_simplex_type()));
    }

    auto pass_through_attributes = options.pass_through_attributes;
    auto other_positions = options.other_position_attributes;

    double length = options.length_abs;
    if (options.length_abs < 0) {
        if (options.length_rel < 0) {
            throw std::runtime_error("Either absolute or relative length must be set!");
        }
        length = relative_to_absolute_length(position, options.length_rel);
    }

    // clear attributes
    std::vector<attribute::MeshAttributeHandle> keeps = pass_through_attributes;
    keeps.emplace_back(position);
    keeps.insert(keeps.end(), other_positions.begin(), other_positions.end());

    // TODO: brig me back!
    // mesh_in->clear_attributes(keeps);

    // gather handles again as they were invalidated by clear_attributes
    // positions = utils::get_attributes(cache, *mesh_in,
    // options.position_attribute); assert(positions.size() == 1); position =
    // positions.front(); pass_through_attributes = utils::get_attributes(cache,
    // *mesh_in, options.pass_through_attributes);

    std::optional<attribute::MeshAttributeHandle> position_for_inversion =
        options.inversion_position_attribute;


    assert(dynamic_cast<TriMesh*>(&position.mesh()) != nullptr);

    TriMesh& mesh = static_cast<TriMesh&>(position.mesh());

    const double length_min = (4. / 5.) * length;
    const double length_max = (4. / 3.) * length;

    std::vector<attribute::MeshAttributeHandle> positions = other_positions;
    positions.push_back(position);

    auto invariant_link_condition =
        std::make_shared<wmtk::invariants::MultiMeshLinkConditionInvariant>(mesh);

    auto invariant_min_edge_length = std::make_shared<MinEdgeLengthInvariant>(
        mesh,
        position.as<double>(),
        length_max * length_max);

    auto invariant_max_edge_length = std::make_shared<MaxEdgeLengthInvariant>(
        mesh,
        position.as<double>(),
        length_min * length_min);

    auto invariant_interior_edge = std::make_shared<invariants::InvariantCollection>(mesh);
    auto invariant_interior_vertex = std::make_shared<invariants::InvariantCollection>(mesh);

    auto set_all_invariants = [&](auto&& m) {
        invariant_interior_edge->add(
            std::make_shared<invariants::InteriorSimplexInvariant>(m, PrimitiveType::Edge));
        invariant_interior_vertex->add(
            std::make_shared<invariants::InteriorSimplexInvariant>(m, PrimitiveType::Vertex));
    };
    multimesh::MultiMeshVisitor visitor(set_all_invariants);
    visitor.execute_from_root(mesh);

    auto invariant_valence_improve =
        std::make_shared<invariants::ValenceImprovementInvariant>(mesh);

    auto invariant_mm_map = std::make_shared<MultiMeshMapValidInvariant>(mesh);

    auto update_position_func = [](const Eigen::MatrixXd& P) -> Eigen::VectorXd {
        return P.col(0);
    };
    std::shared_ptr<wmtk::operations::SingleAttributeTransferStrategy<double, double>>
        update_position;

    if (!options.other_position_attributes.empty()) {
        update_position =
            std::make_shared<wmtk::operations::SingleAttributeTransferStrategy<double, double>>(
                other_positions.front(),
                position,
                update_position_func);
    }

    using namespace operations;

    assert(mesh.is_connectivity_valid());

    // split
    wmtk::logger().debug("Configure isotropic remeshing split");
    auto op_split = std::make_shared<EdgeSplit>(mesh);
    op_split->add_invariant(invariant_min_edge_length);
    if (options.lock_boundary && !options.use_for_periodic && !options.dont_disable_split) {
        op_split->add_invariant(invariant_interior_edge);
    }
    for (auto& p : positions) {
        op_split->set_new_attribute_strategy(
            p,
            SplitBasicStrategy::None,
            SplitRibBasicStrategy::Mean);
    }
    for (const auto& attr : pass_through_attributes) {
        op_split->set_new_attribute_strategy(attr);
    }
    assert(op_split->attribute_new_all_configured());


    //////////////////////////////////////////
    // collapse
    wmtk::logger().debug("Configure isotropic remeshing collapse");
    auto op_collapse = std::make_shared<EdgeCollapse>(mesh);
    op_collapse->add_invariant(invariant_link_condition);
    if (position_for_inversion) {
        op_collapse->add_invariant(std::make_shared<SimplexInversionInvariant<double>>(
            position_for_inversion.value().mesh(),
            position_for_inversion.value().as<double>()));
    }

    op_collapse->add_invariant(invariant_max_edge_length);
    op_collapse->add_invariant(invariant_mm_map);

    // hack for uv
    // if (options.fix_uv_seam) {
    //     op_collapse->add_invariant(
    //         std::make_shared<invariants::uvEdgeInvariant>(mesh, other_positions.front().mesh()));
    // }

    if (options.lock_boundary && !options.use_for_periodic) {
        op_collapse->add_invariant(invariant_interior_edge);
        // set collapse towards boundary
        for (auto& p : positions) {
            auto tmp = std::make_shared<CollapseNewAttributeStrategy<double>>(p);
            tmp->set_strategy(CollapseBasicStrategy::CopyOther);
            tmp->set_simplex_predicate(BasicSimplexPredicate::IsInterior);
            op_collapse->set_new_attribute_strategy(p, tmp);
        }
    } else if (options.use_for_periodic) {
        op_collapse->add_invariant(
            std::make_shared<invariants::FusionEdgeInvariant>(mesh, mesh.get_multi_mesh_root()));
        for (auto& p : positions) {
            op_collapse->set_new_attribute_strategy(p, CollapseBasicStrategy::CopyOther);
        }
    } else {
        for (auto& p : positions) {
            op_collapse->set_new_attribute_strategy(p, CollapseBasicStrategy::CopyOther);
        }
    }


    for (const auto& attr : pass_through_attributes) {
        op_collapse->set_new_attribute_strategy(attr);
    }
    assert(op_collapse->attribute_new_all_configured());


    //////////////////////////////////////////
    // swap
    wmtk::logger().debug("Configure isotropic remeshing swap");
    auto op_swap = std::make_shared<composite::TriEdgeSwap>(mesh);
    op_swap->add_invariant(invariant_interior_edge);

    // hack for uv
    if (options.fix_uv_seam) {
        op_swap->add_invariant(
            std::make_shared<invariants::uvEdgeInvariant>(mesh, other_positions.front().mesh()));
    }

    op_swap->add_invariant(invariant_valence_improve);
    op_swap->collapse().add_invariant(invariant_link_condition);
    op_swap->collapse().add_invariant(invariant_mm_map);
    for (auto& p : positions) {
        op_swap->split().set_new_attribute_strategy(
            p,
            SplitBasicStrategy::None,
            SplitRibBasicStrategy::Mean);
    }
    if (position_for_inversion) {
        op_swap->collapse().add_invariant(std::make_shared<SimplexInversionInvariant<double>>(
            position_for_inversion.value().mesh(),
            position_for_inversion.value().as<double>()));
    }

    for (auto& p : positions)
        op_swap->collapse().set_new_attribute_strategy(p, CollapseBasicStrategy::CopyOther);
    for (const auto& attr : pass_through_attributes) {
        op_swap->split().set_new_attribute_strategy(attr);
        op_swap->collapse().set_new_attribute_strategy(attr);
    }
    assert(op_swap->split().attribute_new_all_configured());
    assert(op_swap->collapse().attribute_new_all_configured());


    //////////////////////////////////////////
    // smooth
    auto op_smooth = std::make_shared<AttributesUpdateWithFunction>(mesh);
    if (position.dimension() == 3) {
        op_smooth->set_function(VertexTangentialLaplacianSmooth(position));
    } else {
        op_smooth->set_function(VertexLaplacianSmooth(position));
    }

    if (options.lock_boundary) {
        op_smooth->add_invariant(invariant_interior_vertex);
    }

    // hack for uv
    if (options.fix_uv_seam) {
        op_smooth->add_invariant(
            std::make_shared<invariants::uvEdgeInvariant>(mesh, other_positions.front().mesh()));
    }

    if (position_for_inversion) {
        op_smooth->add_invariant(std::make_shared<SimplexInversionInvariant<double>>(
            position_for_inversion.value().mesh(),
            position_for_inversion.value().as<double>()));
    }

    if (update_position) op_smooth->add_transfer_strategy(update_position);


    //////////////////////////////////////////
    Scheduler scheduler;
    for (long i = 0; i < options.iterations; ++i) {
        wmtk::logger().info("Iteration {}", i);

        SchedulerStats pass_stats;
        pass_stats += run_reference_split(*op_split, position, length_max, options.lock_boundary);

        pass_stats += run_reference_collapse(
            *op_collapse,
            position,
            length_min,
            length_max,
            options.lock_boundary);
        pass_stats += run_reference_swap(*op_swap, options.lock_boundary);
        pass_stats += scheduler.run_operation_on_all(*op_smooth);

        auto op_consolidate = MeshConsolidate(mesh);
        op_consolidate(simplex::Simplex(mesh, PrimitiveType::Vertex, Tuple()));
        // multimesh::consolidate(mesh);

        logger().info(
            "Executed {} ops (S/F) {}/{}. Time: collecting: {}, sorting: {}, executing: {}",
            pass_stats.number_of_performed_operations(),
            pass_stats.number_of_successful_operations(),
            pass_stats.number_of_failed_operations(),
            pass_stats.collecting_time,
            pass_stats.sorting_time,
            pass_stats.executing_time);

        // multimesh::consolidate(mesh);
    }
}
} // namespace wmtk::components::isotropic_remeshing

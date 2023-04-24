#include <wmtk/ExecutionScheduler.hpp>
#include "ExtremeOpt.h"
#include "SYMDIR.h"
#include "SYMDIR_NEW.h"
namespace {

auto renew = [](auto& m, auto op, auto& tris) {
    auto edges = m.new_edges_after(tris);
    auto optup = std::vector<std::pair<std::string, wmtk::TriMesh::Tuple>>();
    for (auto& e : edges) optup.emplace_back(op, e);
    return optup;
};
using namespace extremeopt;
using namespace wmtk;
class ExtremeOptSwapEdgeOperation : public wmtk::TriMeshOperationShim<
                                        ExtremeOpt,
                                        ExtremeOptSwapEdgeOperation,
                                        wmtk::TriMeshSwapEdgeOperation>
{
public:
    ExecuteReturnData execute(ExtremeOpt& m, const Tuple& t)
    {
        return wmtk::TriMeshSwapEdgeOperation::execute(m, t);
    }
    bool before(ExtremeOpt& m, const Tuple& t)
    {
        if (wmtk::TriMeshSwapEdgeOperation::before(m, t)) {
            return m.swap_edge_before(t);
        }
        return false;
    }

    bool after(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= wmtk::TriMeshSwapEdgeOperation::after(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.swap_edge_after(ret_data.tuple);
        }
        return ret_data;
    }
    bool invariants(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= wmtk::TriMeshSwapEdgeOperation::invariants(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.invariants(ret_data.new_tris);
        }
        return ret_data;
    }
};

template <typename Executor>
void addCustomOps(Executor& e)
{
    e.add_operation(std::make_shared<ExtremeOptSwapEdgeOperation>());
}
} // namespace
bool extremeopt::ExtremeOpt::swap_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) return false;

    // std::cout << "trying to swap edge " << t.vid(*this) << "," << t.switch_vertex(*this).vid(*this) << std::endl;

    Tuple t1 = t.switch_vertex(*this).switch_edge(*this);
    Tuple t2 = t.switch_face(*this).value().switch_edge(*this);
    if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
    if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);
    swap_cache.local().t1 = t1;
    swap_cache.local().t2 = t2;

    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
    double E = E_eval.symmetric_dirichlet_energy_2chart(*this, t);
    swap_cache.local().E_old = E;

    return true;
    // std::cout << "energy before swap is " << swap_cache.local().E_old << std::endl;
}


bool extremeopt::ExtremeOpt::swap_edge_after(const Tuple& t)
{
    // std::cout << "after swapping the edge becomes " << t.vid(*this) << "," << t.switch_vertex(*this).vid(*this) << std::endl;

    auto tri1 = oriented_tri_vids(t);
    auto t_opp = t.switch_face(*this);
    auto tri2 = oriented_tri_vids(t_opp.value());
    if (is_inverted(t) || is_inverted(t_opp.value()))
    {
        return false;
    }
    if (is_3d_degenerated(t) || is_3d_degenerated(t_opp.value()))
    {
        return false;
    }

    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
    double E = E_eval.symmetric_dirichlet_energy_2chart(*this, t);
    
    if (!std::isfinite(E) || E >= swap_cache.local().E_old) {
        return false;
    }
    // std::cout << "Energy after swapping is " << E << std::endl;
    // std::cout << std::endl;

    if (m_params.with_cons) {
        // update constraints after swap
        Tuple t1 = t.switch_edge(*this);
        Tuple t2 = t.switch_face(*this).value().switch_vertex(*this).switch_edge(*this);
        Tuple t1_old = swap_cache.local().t1;
        Tuple t2_old = swap_cache.local().t2;
        if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
        if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);
        bool flag = true;
        if (is_boundary_edge(t1)) {
            auto t1_old_pair = edge_attrs[t1_old.eid_unsafe(*this)].pair;
            if (t1_old_pair.eid_unsafe(*this) == t2_old.eid_unsafe(*this)) {
                flag = false;
                edge_attrs[t1.eid(*this)].pair = t2;
                edge_attrs[t2.eid(*this)].pair = t1;
            } else {
                edge_attrs[t1_old_pair.eid_unsafe(*this)].pair = t1;
                edge_attrs[t1.eid(*this)].pair = t1_old_pair;
            }
        }
        if (flag && is_boundary_edge(t2)) {
            auto t2_old_pair = edge_attrs[t2_old.eid_unsafe(*this)].pair;
            edge_attrs[t2_old_pair.eid_unsafe(*this)].pair = t2;
            edge_attrs[t2.eid(*this)].pair = t2_old_pair;
        }

        auto one_ring_tris = get_one_ring_tris_for_vertex(t);

        for (auto t_tmp : one_ring_tris) {
            Tuple t0 = t_tmp;
            Tuple t1 = t0.switch_edge(*this);
            Tuple t2 = t1.switch_vertex(*this).switch_edge(*this);
            if (!t0.is_ccw(*this)) t0 = t0.switch_vertex(*this);
            if (!t1.is_ccw(*this)) t1 = t1.switch_vertex(*this);
            if (!t2.is_ccw(*this)) t2 = t2.switch_vertex(*this);

            if (is_boundary_edge(t0)) {
                edge_attrs[edge_attrs[t0.eid(*this)].pair.eid_unsafe(*this)].pair = t0;
            }
            if (is_boundary_edge(t1)) {
                edge_attrs[edge_attrs[t1.eid(*this)].pair.eid_unsafe(*this)].pair = t1;
            }
            if (is_boundary_edge(t2)) {
                edge_attrs[edge_attrs[t2.eid(*this)].pair.eid_unsafe(*this)].pair = t2;
            }
        }
    }


    return true;
}

void extremeopt::ExtremeOpt::swap_all_edges()
{
    auto collect_all_ops_swap = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_edges()) {
        collect_all_ops_swap.emplace_back("edge_swap", loc);
    }

    auto setup_and_execute = [&](auto& executor_swap) {
        addCustomOps(executor_swap);
        executor_swap.renew_neighbor_tuples = renew;
        executor_swap.priority = [&](auto& m, auto _, auto& e) {
            return -(vertex_attrs[e.vid(*this)].pos -
                     vertex_attrs[e.switch_vertex(*this).vid(*this)].pos)
                        .norm();
        };
        executor_swap.num_threads = NUM_THREADS;
        executor_swap(*this, collect_all_ops_swap);
    };
    auto executor_swap = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    setup_and_execute(executor_swap);
}

#include "ExtremeOpt.h"
#include <wmtk/ExecutionScheduler.hpp>
#include "SYMDIR.h"
#include "SYMDIR_NEW.h"
namespace {

using namespace extremeopt;
using namespace wmtk;

class ExtremeOptSplitEdgeOperation : public wmtk::TriMeshOperationShim<
                                                  ExtremeOpt,
                                                  ExtremeOptSplitEdgeOperation,
                                                  wmtk::TriMeshSplitEdgeOperation>
{
public:
    ExecuteReturnData execute(ExtremeOpt& m, const Tuple& t)
    {
        return wmtk::TriMeshSplitEdgeOperation::execute(m, t);
    }
    bool before(ExtremeOpt& m, const Tuple& t)
    {
        if (wmtk::TriMeshSplitEdgeOperation::before(m, t)) {
            return  m.split_edge_before(t);
        }
        return false;
    }
    bool after(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {   
        ret_data.success &= wmtk::TriMeshSplitEdgeOperation::after(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.split_edge_after(ret_data.tuple);
        }
        return ret_data;
    }
    bool invariants(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= wmtk::TriMeshSplitEdgeOperation::invariants(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.invariants(ret_data.new_tris);
        }
        return ret_data;
    }
};

    template <typename Executor>
    void addCustomOps(Executor& e) {

        e.add_operation(std::make_shared<ExtremeOptSplitEdgeOperation>());
    }
} // namespace

bool extremeopt::ExtremeOpt::split_edge_before(const Tuple& t)
{
    if (!t.is_valid(*this)) {
        return false;
    }

    if (is_boundary_edge(t)) {
        return false;
    }

    position_cache.local().V1 = vertex_attrs[t.vid(*this)].pos_3d;
    position_cache.local().V2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos_3d;
    position_cache.local().uv1 = vertex_attrs[t.vid(*this)].pos;
    position_cache.local().uv2 = vertex_attrs[t.switch_vertex(*this).vid(*this)].pos;

    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
    position_cache.local().E_before = E_eval.symmetric_dirichlet_energy_2chart(*this, t);
    return true;
}

bool extremeopt::ExtremeOpt::split_edge_after(const Tuple& t)
{
    Eigen::Vector3d V = (position_cache.local().V1 + position_cache.local().V2) / 2.0;
    Eigen::Vector2d uv = (position_cache.local().uv1 + position_cache.local().uv2) / 2.0;
    Tuple vert_tuple = t.switch_vertex(*this);
    size_t vid = vert_tuple.vid(*this);

    if (m_params.do_projection)
    {
        Eigen::VectorXd sqrD;
        int fid;
        Eigen::RowVector3d C;
        tree.squared_distance(input_V, input_F, Eigen::RowVector3d(V), fid, C);
        V = C;
    }

    // update the attributes
    auto& v = vertex_attrs[vid];
    v.pos = uv;
    v.pos_3d = V;

    // check for inverted/degenerate triangles
    auto locs = get_one_ring_tris_for_vertex(vert_tuple);
    for (auto loc : locs)
    {
        if (is_inverted(loc) || is_3d_degenerated(loc))
        {
            return false;
        }
    }

    wmtk::SymmetricDirichletEnergy E_eval(wmtk::SymmetricDirichletEnergy::EnergyType::Lp, m_params.Lp);
    double E = E_eval.symmetric_dirichlet_energy_onering(*this, vert_tuple);

    if (!std::isfinite(E))
    {
        return false;
    }

    // if use projection, we need to check Energy
    if (m_params.do_projection)
    {
        if (E > position_cache.local().E_before)
        {
            return false;
        }
    }

    // avoid degenrate triangles
    for (size_t nbr_vid : get_one_ring_vids_for_vertex(vid)) {
        if (nbr_vid != vid && vertex_attrs[nbr_vid].pos == v.pos) {
            return false;
        }
    }


    return true;
}

void extremeopt::ExtremeOpt::split_all_edges(const Eigen::VectorXd &Es)
{
    Eigen::MatrixXi EE;
    if (m_params.with_cons)
    {
        export_EE(EE);
    }
    size_t vid_threshold = 0;
    auto collect_all_ops_split = std::vector<std::pair<std::string, Tuple>>();

    for (auto& loc : get_faces())
    {
        auto t0 = loc;
        auto t1 = t0.switch_edge(*this);
        auto t2 = t0.switch_vertex(*this).switch_edge(*this);
        double l0 =
            (vertex_attrs[t0.vid(*this)].pos - vertex_attrs[t0.switch_vertex(*this).vid(*this)].pos)
                .norm();
        double l1 =
            (vertex_attrs[t1.vid(*this)].pos - vertex_attrs[t1.switch_vertex(*this).vid(*this)].pos)
                .norm();
        double l2 =
            (vertex_attrs[t2.vid(*this)].pos - vertex_attrs[t2.switch_vertex(*this).vid(*this)].pos)
                .norm();
        if (is_boundary_edge(t0)) l0 = 0;
        if (is_boundary_edge(t1)) l1 = 0;
        if (is_boundary_edge(t2)) l2 = 0;

        if (l0 >= l1 && l0 >= l2) {
            collect_all_ops_split.emplace_back("edge_split", t0);
        } else if (l1 >= l2) {
            collect_all_ops_split.emplace_back("edge_split", t1);

        } else {
            collect_all_ops_split.emplace_back("edge_split", t2);
        }
    }
    
    
    auto setup_and_execute = [&](auto& executor_split) {
        addCustomOps(executor_split);
        executor_split.priority = [&](auto& m, auto _, auto& e) {
            if (e.fid(*this) >= Es.size()) return 1e50;
            else return Es(e.fid(*this));
        };
        executor_split.stopping_criterion_checking_frequency = 300;
        executor_split.stopping_criterion = [](const TriMesh&) {
            return true; // non-stop, process everything
        };
        executor_split.num_threads = NUM_THREADS;
        executor_split(*this, collect_all_ops_split);
    };
    auto executor_split = wmtk::ExecutePass<ExtremeOpt, wmtk::ExecutionPolicy::kSeq>();
    setup_and_execute(executor_split);
    
    if (m_params.with_cons)
    {
        update_constraints_EE_v(EE);
    }
}

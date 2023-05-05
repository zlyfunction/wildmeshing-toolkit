#include "ExtremeOpt.h"
#include <wmtk/ExecutionScheduler.hpp>
#include "SYMDIR.h"
#include "SYMDIR_NEW.h"
namespace {

using namespace extremeopt;
using namespace wmtk;


class SplitPairOperation : public wmtk::TriMeshOperationShim<ExtremeOpt, SplitPairOperation>
{
public:
    bool before(ExtremeOpt& m, const TriMesh::Tuple& t)
    {
        if (!t.is_valid(m)) {
            return false;
        }
        Tuple& t_pair_input = t_pair_input_per_thread.local();
        t_pair_input = m.edge_attrs[t.eid(m)].pair;
        
        if (t.fid(m) == t_pair_input.fid(m))
        {
            return false;
        }

        V1 = m.vertex_attrs[t.vid(m)].pos_3d;
        V2 = m.vertex_attrs[t.switch_vertex(m).vid(m)].pos_3d;
        uv1 = m.vertex_attrs[t.vid(m)].pos;
        uv2 = m.vertex_attrs[t.switch_vertex(m).vid(m)].pos;
        
        V1_pair = m.vertex_attrs[t_pair_input.vid(m)].pos_3d;
        V2_pair = m.vertex_attrs[t_pair_input.switch_vertex(m).vid(m)].pos_3d;
        uv1_pair = m.vertex_attrs[t_pair_input.vid(m)].pos;
        uv2_pair = m.vertex_attrs[t_pair_input.switch_vertex(m).vid(m)].pos;

        return true;
    }

    bool after(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {   
        std::cout << "in after check of split_pair" << std::endl;
        const TriMesh::Tuple& new_t = ret_data.tuples[0];
        bool flag1 = m.split_bd_edge_after(V1, V2, uv1, uv2, new_t);
        if (!flag1)
        {
            std::cout << "fail in split t" << std::endl;
            return false;
        }
        const TriMesh::Tuple& new_t_pair = ret_data.tuples[1];
        bool flag2 = m.split_bd_edge_after(V1_pair, V2_pair, uv1_pair, uv2_pair, new_t_pair);
        if (!flag2)
        {
            std::cout << "fail in split t_pair" << std::endl;
        }

        std::cout << "split_pair successful" << std::endl;
        return true;
    }

    ExecuteReturnData execute(ExtremeOpt& m, const Tuple& t)
    {
        std::cout << "trying to split pair" << std::endl;
        ExecuteReturnData ret_data;
        const Tuple& t_pair_input = t_pair_input_per_thread.local();
        std::cout << t.eid(m) << "\t" << t_pair_input.eid(m) << std::endl;
        auto ret1 = m_edge_spliter.execute(m, t);
        Tuple t_pair =
            m.tuple_from_edge(t_pair_input.eid_unsafe(m) / 3, t_pair_input.eid_unsafe(m) % 3);
        auto ret2 = m_edge_spliter.execute(m, t_pair);
        ret_data.combine(ret1);
        ret_data.combine(ret2);

        ret_data.tuples.push_back(ret1.tuple);
        ret_data.tuples.push_back(ret2.tuple);
        std::cout << "finish topology part of the split_pair" << std::endl;
        std::cout << ret1.success << "\t" << ret2.success << "\t" << ret_data.success << std::endl;
        return ret_data;
    }

    std::string name() const override { return "split_pair"; }
    SplitPairOperation(){};
    virtual ~SplitPairOperation(){};

    tbb::enumerable_thread_specific<Tuple> t_pair_input_per_thread;
    wmtk::TriMeshSplitEdgeOperation m_edge_spliter;
    Eigen::Vector3d V1,V1_pair;
    Eigen::Vector3d V2,V2_pair;
    Eigen::Vector2d uv1,uv1_pair;
    Eigen::Vector2d uv2,uv2_pair;
};

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
        e.add_operation(std::make_shared<SplitPairOperation>());
        e.add_operation(std::make_shared<ExtremeOptSplitEdgeOperation>());
    }
} // namespace

bool extremeopt::ExtremeOpt::split_edge_before(const Tuple& t)
{   

    // DEBUG_FID
    if (t.fid(*this) == 26198)
    {
        std::cout << "face 26198 for split!" << std::endl;
    }

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

bool extremeopt::ExtremeOpt::split_bd_edge_after(Eigen::Vector3d V1, Eigen::Vector3d V2, Eigen::Vector2d uv1, Eigen::Vector2d uv2, const Tuple& t)
{
    Eigen::Vector3d V = (V1 + V2) / 2.0;
    Eigen::Vector2d uv = (uv1 + uv2) / 2.0;
    Tuple vert_tuple = t.switch_vertex(*this);
    size_t vid = vert_tuple.vid(*this);

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

        if (l0 >= l1 && l0 >= l2) {
            if (is_boundary_edge(t0))
            {
                collect_all_ops_split.emplace_back("split_pair", t0);
            }
            else
                collect_all_ops_split.emplace_back("edge_split", t0);
        } else if (l1 >= l2) {
            if (is_boundary_edge(t1))
            {
                collect_all_ops_split.emplace_back("split_pair", t1);
            }
            else
                collect_all_ops_split.emplace_back("edge_split", t1);

        } else {
            if (is_boundary_edge(t2))
            {
                collect_all_ops_split.emplace_back("split_pair", t2);
            }
            else
                collect_all_ops_split.emplace_back("edge_split", t2);
        }
    }
    
    
    auto setup_and_execute = [&](auto& executor_split) {
        addCustomOps(executor_split);
        executor_split.priority = [&](auto& m, auto _, auto& e) {
            if (e.fid(*this) >= Es.size()) return 1e50;
            else return Es(e.fid(*this));
        };
        executor_split.stopping_criterion_checking_frequency = m_params.split_succ_cnt;
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

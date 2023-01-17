#pragma once

#include <igl/PI.h>
#include <igl/Timer.h>
#include <wmtk/TriMesh.h>
#include <wmtk/TriMeshOperation.h>
#include "Parameters.h"
#include "json.hpp"
using json = nlohmann::json;

namespace extremeopt {

class VertexAttributes
{
public:
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    Eigen::Vector3d pos_3d = Eigen::Vector3d::Zero();

    size_t partition_id = 0; // TODO this should not be here

    // Vertices marked as fixed cannot be modified by any local operation
    bool fixed = false;
};


class FaceAttributes
{
public:
    double area_3d;
};

class EdgeAttributes
{
public:
    wmtk::TriMesh::Tuple pair;
};

class ExtremeOpt : public wmtk::TriMesh
{
public:

Parameters m_params;
// Energy Assigned to undefined energy
// TODO: why not the max double?
const double MAX_ENERGY = 1e50;


    double elen_threshold;
    double elen_threshold_3d;

    class CollapsePair : public wmtk::TriMeshOperation
    {
    public:
        bool before(const TriMesh::Tuple& t, ExtremeOpt& m)
        {
            const bool val = before_check(t, m);
            if (val) {
                m.start_protected_connectivity();
            }
            return val;
        }

    bool after(const TriMesh::Tuple& t, ExtremeOpt& m, std::vector<TriMesh::Tuple> &new_tris)
    {
        m.start_protected_attributes();
        const bool val = after_check(t, m);
        if (!val || !m.invariants(new_tris)) {
            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
            return false;
        }
        m.release_protected_connectivity();
        m.release_protected_attributes();
        return true;
    }

    std::pair<Tuple, bool>  execute(const Tuple& t, ExtremeOpt& m, std::vector<Tuple> &new_tris)
    {
            // TODO: Relocate this code in before check
            if (!m.is_boundary_edge(t)) {
                // std::cout << "not boundary edge" << std::endl;
                return {{}, false};
            }
            if (!t.is_valid(m)) {
                std::cout << "not valid" << std::endl;
                return {{}, false};
            }
            if (!m.wmtk::TriMesh::collapse_edge_before(t)) {
                // std::cout << "link condition error" << std::endl;
                return {{}, false};
            }
            Tuple t_pair_input = m.edge_attrs[t.eid(m)].pair;
            if (!m.wmtk::TriMesh::collapse_edge_before(t_pair_input)) {
                // std::cout << "link condition error" << std::endl;
                return {{}, false};
            }
            // Skip cases that paired edges are in the same triangle
            if (t_pair_input.fid(m) == t.fid(m)) {
                return {{}, false};
            }
            if (t_pair_input.vid(m) == t.switch_vertex(m).vid(m)) {
                return {{}, false};
            }
            if (t_pair_input.switch_vertex(m).vid(m) == t.vid(m)) {
                return {{}, false};
            }
        // Get E_max before collapse
        double E_max_t_input = std::max(m.get_e_max_onering(t), m.get_e_max_onering(t.switch_vertex(m)));
        double E_max_t_pair_input = std::max(m.get_e_max_onering(t_pair_input), m.get_e_max_onering(t_pair_input.switch_vertex(m)));
        double E_max_input = std::max(E_max_t_input, E_max_t_pair_input);
        // std::cout << "trying to collapse a boudnary edge" << std::endl;
        // t.print_info();
        // t_pair_input.print_info();
        // std::cout << "E_max before collapsing is " << E_max_input << std::endl;

        // get neighbor edges
        auto onering_t_l = m.get_one_ring_edges_for_vertex(t);
        auto onering_t_r = m.get_one_ring_edges_for_vertex(t.switch_vertex(m));
        Tuple bd_t_l, bd_t_r;
        for (auto t_tmp : onering_t_l)
        {
            if (m.is_boundary_edge(t_tmp))
            {
                if (t_tmp.eid(m) != t.eid(m))
                {
                    bd_t_l = t_tmp.is_ccw(m)?t_tmp:t_tmp.switch_vertex(m);
                }
            }
        }
        for (auto t_tmp : onering_t_r)
        {
            if (m.is_boundary_edge(t_tmp))
            {
                if (t_tmp.eid(m) != t.eid(m))
                {
                    bd_t_r = t_tmp.is_ccw(m)?t_tmp:t_tmp.switch_vertex(m);
                }
            }
        }

        Tuple bd_t_l_pair = m.edge_attrs[bd_t_l.eid(m)].pair;
        Tuple bd_t_r_pair = m.edge_attrs[bd_t_r.eid(m)].pair;
        bool keep_t = false, keep_t_opp = false;
        if (bd_t_r_pair.switch_vertex(m).vid(m) == t_pair_input.vid(m))
        {   
            auto len1 = (m.vertex_attrs[t.vid(m)].pos - m.vertex_attrs[bd_t_r.switch_vertex(m).vid(m)].pos).norm();
            auto len2 = (m.vertex_attrs[t_pair_input.switch_vertex(m).vid(m)].pos - m.vertex_attrs[bd_t_r_pair.vid(m)].pos).norm();
            if (std::abs(len1 - len2) < 1e-7)
            {
                // std::cout << "keep t.vid" << std::endl;
                keep_t = true;
            }
            else
            {
                // std::cout << "len diff, cannot keep t.vid" << std::endl;
            }
        }
        else
        {
            // std::cout << "cannot keep t.vid" << std::endl;
        }

        if (bd_t_l_pair.vid(m) == t_pair_input.switch_vertex(m).vid(m))
        {
            auto len1 = (m.vertex_attrs[t.switch_vertex(m).vid(m)].pos - m.vertex_attrs[bd_t_l.vid(m)].pos).norm();
            auto len2 = (m.vertex_attrs[t_pair_input.vid(m)].pos - m.vertex_attrs[bd_t_l_pair.switch_vertex(m).vid(m)].pos).norm();
            if (std::abs(len1 - len2) < 1e-7)
            {
                // std::cout << "keep t.switch_vertex.vid" << std::endl;
                keep_t_opp = true;
            }
            else
            {
                // std::cout << "len diff, cannot keep t.switch_vertex.vid" << std::endl;
            }
        }
        else
        {
            // std::cout << "cannot keep t.switch_vertex.vid" << std::endl;
        }
        if (!keep_t && !keep_t_opp)
        {
            // std::cout << "this boudnary edge cannot collapse" << std::endl;
            return {{}, false};
        }
        Eigen::Vector3d V_keep_t, V_keep_t_pair;
        Eigen::Vector2d uv_keep_t, uv_keep_t_pair;
        if (keep_t)
        {
            V_keep_t = m.vertex_attrs[t.vid(m)].pos_3d;
            uv_keep_t = m.vertex_attrs[t.vid(m)].pos;
            V_keep_t_pair = m.vertex_attrs[t_pair_input.switch_vertex(m).vid(m)].pos_3d;
            uv_keep_t_pair = m.vertex_attrs[t_pair_input.switch_vertex(m).vid(m)].pos;
        }
        else
        {
            V_keep_t = m.vertex_attrs[t.switch_vertex(m).vid(m)].pos_3d;
            uv_keep_t = m.vertex_attrs[t.switch_vertex(m).vid(m)].pos;
            V_keep_t_pair = m.vertex_attrs[t_pair_input.vid(m)].pos_3d;
            uv_keep_t_pair = m.vertex_attrs[t_pair_input.vid(m)].pos;
        }

        m.start_protected_connectivity();
        m.start_protected_attributes();
        auto new_t = m.collapse_edge_new(t, new_tris);
        
        
        double E_max_t, E_max_t_pair;
        if (!m.collapse_bd_edge_after(new_t, V_keep_t, uv_keep_t, bd_t_l, bd_t_r, E_max_t))
        {
            // std::cout << "collapse t fail" << std::endl;
            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
            return {{}, false};
        }
        else
        {
            // std::cout << "collapse t ok" << std::endl;
        }

        Tuple t_pair = m.tuple_from_edge(t_pair_input.eid_unsafe(m) / 3, t_pair_input.eid_unsafe(m) % 3);
        auto onering_t_pair_l = m.get_one_ring_edges_for_vertex(t_pair);
        auto onering_t_pair_r = m.get_one_ring_edges_for_vertex(t_pair.switch_vertex(m));
        Tuple bd_t_pair_l, bd_t_pair_r;
        for (auto t_tmp : onering_t_pair_l)
        {
            if (m.is_boundary_edge(t_tmp))
            {
                if (t_tmp.eid(m) != t_pair.eid(m))
                {
                    bd_t_pair_l = t_tmp.is_ccw(m)?t_tmp:t_tmp.switch_vertex(m);
                }
            }
        }
        for (auto t_tmp : onering_t_pair_r)
        {
            if (m.is_boundary_edge(t_tmp))
            {
                if (t_tmp.eid(m) != t_pair.eid(m))
                {
                    bd_t_pair_r = t_tmp.is_ccw(m)?t_tmp:t_tmp.switch_vertex(m);
                }
            }
        }
        new_t = m.collapse_edge_new(t_pair, new_tris);
        if (!m.collapse_bd_edge_after(new_t, V_keep_t_pair, uv_keep_t_pair, bd_t_pair_l, bd_t_pair_r, E_max_t_pair))
        {
            // std::cout << "collapse t pair fail" << std::endl;
            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
            return {{}, false};
        }
        else
        {
            // std::cout << "collapse t pair ok" << std::endl;
        }
        if (E_max_input < std::max(E_max_t, E_max_t_pair))
        {
            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
            return {{}, false};
        }
        // std::cout << "good!" << std::endl;
        m.release_protected_connectivity();
        m.release_protected_attributes();

                
        return {new_t,true};
    }

        bool before_check(const Tuple& t, ExtremeOpt& m) { return m.collapse_edge_before(t); }

        bool after_check(const Tuple& t, ExtremeOpt& m) { return m.collapse_edge_after(t); }

        ExecuteReturnData execute(const Tuple& t, TriMesh& m) override
        {
            ExecuteReturnData ret_data;
            std::vector<TriMesh::Tuple> new_tris;
            if (std::tie(ret_data.tuple, ret_data.success) =
                    execute(t, dynamic_cast<ExtremeOpt&>(m), ret_data.new_tris);
                ret_data.success) {
                return ret_data;
            } else {
                return {};
            }
        }
        bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) override
        {
            return after_check(ret_data.tuple, dynamic_cast<ExtremeOpt&>(m));
        }
        bool before_check(const Tuple& t, TriMesh& m) override
        {
            return before_check(t, dynamic_cast<ExtremeOpt&>(m));
        }
        std::string name() const { return "test_op"; }
        CollapsePair(){};
        virtual ~CollapsePair(){};
    };

    ExtremeOpt(){};

    virtual ~ExtremeOpt(){};

// Store the per-vertex and per-face attributes
wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
wmtk::AttributeCollection<FaceAttributes> face_attrs;
wmtk::AttributeCollection<EdgeAttributes> edge_attrs;

void consolidate_mesh_cons();

bool check_constraints(double eps = 1e-7);

struct PositionInfoCache
{
    int vid1;
    int vid2;
    Eigen::Vector3d V1;
    Eigen::Vector3d V2;
    Eigen::Vector2d uv1;
    Eigen::Vector2d uv2;
    bool is_v1_bd;
    bool is_v2_bd;
    Tuple bd_e1;
    Tuple bd_e2;
    double E_max_before_collpase;
};
tbb::enumerable_thread_specific<PositionInfoCache> position_cache;

tbb::enumerable_thread_specific<std::pair<Tuple, Tuple>> swap_cache;
// Initializes the mesh
void create_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv);

// Initialize the mesh with constraints
void init_constraints(const std::vector<std::vector<int>> &EE_e);
void update_constraints_EE_v(const Eigen::MatrixXi &EE);

// Exports V and F of the stored mesh
void export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv);

// Export constraints EE
void export_EE(Eigen::MatrixXi &EE);

// Writes a triangle mesh in OBJ format
void write_obj(const std::string& path);

// Computes the quality of a triangle
double get_quality(const Tuple& loc) const;

// Computes the average quality of a mesh
Eigen::VectorXd get_quality_all_triangles();

// compute the max_E of a one ring
void get_mesh_onering(const Tuple& t, Eigen::MatrixXd &V_local, Eigen::MatrixXd &uv_local, Eigen::MatrixXi &F_local);
double get_e_max_onering(const Tuple &t);

// Check if a triangle is inverted
bool is_inverted(const Tuple& loc) const;

// Optimization
void do_optimization(json &opt_log);

// Vertex Smoothing
bool smooth_before(const Tuple& t) override;
bool smooth_after(const Tuple& t) override;
void smooth_all_vertices();
void smooth_global(int steps);

// Edge Swapping
std::vector<wmtk::TriMesh::Tuple> new_edges_after(const std::vector<wmtk::TriMesh::Tuple>& tris) const;
std::vector<wmtk::TriMesh::Tuple> replace_edges_after_split(const std::vector<wmtk::TriMesh::Tuple>& tris, const size_t vid_threshold) const;

bool swap_edge_before(const Tuple& t) override;
bool swap_edge_after(const Tuple& t) override;
void swap_all_edges();

// Edge Collapsing
void cache_edge_positions(const Tuple& t);
bool collapse_edge_before(const Tuple& t) override;
bool collapse_edge_after(const Tuple& t) override;
void collapse_all_edges();
bool collapse_bd_edge_after(const Tuple& t, const Eigen::Vector3d &V_keep, const Eigen::Vector2d &uv_keep,Tuple &t_l_old, Tuple &t_r_old, double &E_max);
// Edge Splitting
bool split_edge_before(const Tuple& t) override;
bool split_edge_after(const Tuple& t) override;
void split_all_edges();
};

} // namespace extremeopt

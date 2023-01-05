#pragma once

#include <igl/Timer.h>
#include <igl/PI.h>
#include <wmtk/TriMesh.h>
#include "Parameters.h"
#include "json.hpp"
using json = nlohmann::json;

namespace extremeopt {

class VertexAttributes
{
public:
    Eigen::Vector2d pos; 
    Eigen::Vector3d pos_3d;

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

class CollapsePair:public wmtk::TriMesh::Operation
{
public:
    bool before(const TriMesh::Tuple& t, ExtremeOpt& m)
    {
        const bool val = before_check(t, m);
        if (val) {
            m.start_protect_connectivity();
        }
        return val;
    }

    bool after(const TriMesh::Tuple& t, ExtremeOpt& m, std::vector<TriMesh::Tuple> &new_tris)
    {
        m.start_protect_attributes();
        const bool val = after_check(t, m);
        if (!val || !m.invariants(new_tris)) {
            m.rollback_protected_connectivity();
            m.rollback_protected_attributes();
            return false;
        }
        m.release_protect_connectivity();
        m.release_protect_attributes();
        return true;
    }

    bool execute(const Tuple& t, ExtremeOpt& m, std::vector<Tuple> &new_tris)
    {
        return false;
        // TODO: Relocate this code
        if (!t.is_valid(m))
        {
            std::cout << "not valid" << std::endl;
            return false;
        }
        if (!m.wmtk::TriMesh::collapse_edge_before(t))
        {
            std::cout << "link condition error" << std::endl;
            return false;
        }

        Tuple t_pair_input = m.edge_attrs[t.eid(m)].pair;
        // Skip cases that paired edges are in the same triangle
        if (t_pair_input.fid(m) == t.fid(m))
        {
            return false;
        }

        // Get E_max before collapse
        double E_max_t_input = std::max(m.get_e_max_onering(t), m.get_e_max_onering(t.switch_vertex(m)));
        double E_max_t_pair_input = std::max(m.get_e_max_onering(t_pair_input), m.get_e_max_onering(t_pair_input.switch_vertex(m)));
        double E_max_input = std::max(E_max_t_input, E_max_t_pair_input);

        std::cout << "trying to collapse a boudnary edge" << std::endl;
        std::cout << "E_max before collapsing is " << E_max_input << std::endl;

        // get neighbor edges
        auto onering_t_l = m.get_one_ring_edges_for_vertex(t);
        auto onering_t_r = m.get_one_ring_edges_for_vertex(t.switch_vertex(m));

        
        return false;
        if (before(t, m))
        {
            auto new_t = m.collapse_edge_new(t, new_tris);
            return after(new_t, m, new_tris);
        }
        else
        {
            return false;
        }
    }

    bool before_check(const Tuple& t, ExtremeOpt& m) 
    {
        return m.collapse_edge_before(t); 
    }

    bool after_check(const Tuple& t, ExtremeOpt& m) 
    {
        return m.collapse_edge_after(t);
    }
    CollapsePair() {};
    virtual ~CollapsePair(){};
};

ExtremeOpt() {};

virtual ~ExtremeOpt() {};


// Store the per-vertex and per-face attributes
wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
wmtk::AttributeCollection<FaceAttributes> face_attrs;
wmtk::AttributeCollection<EdgeAttributes> edge_attrs;

void consolidate_mesh_cons()
{
    auto v_cnt = 0;
    std::vector<size_t> map_v_ids(vert_capacity(), -1);
    for (auto i = 0; i < vert_capacity(); i++) {
        if (m_vertex_connectivity[i].m_is_removed) continue;
        map_v_ids[i] = v_cnt;
        v_cnt++;
    }

    auto t_cnt = 0;
    std::vector<size_t> map_t_ids(tri_capacity(), -1);
    for (auto i = 0; i < tri_capacity(); i++) {
        if (m_tri_connectivity[i].m_is_removed) continue;
        map_t_ids[i] = t_cnt;
        t_cnt++;
    }
    v_cnt = 0;
    for (auto i = 0; i < vert_capacity(); i++) {
        if (m_vertex_connectivity[i].m_is_removed) continue;
        if (v_cnt != i) {
            assert(v_cnt < i);
            m_vertex_connectivity[v_cnt] = m_vertex_connectivity[i];
            if (p_vertex_attrs) p_vertex_attrs->move(i, v_cnt);
        }
        for (size_t& t_id : m_vertex_connectivity[v_cnt].m_conn_tris) t_id = map_t_ids[t_id];
        v_cnt++;
    }
    t_cnt = 0;

    for (int i = 0; i < tri_capacity(); i++) {
        if (m_tri_connectivity[i].m_is_removed) continue;

        if (t_cnt != i) {
            assert(t_cnt < i);
            m_tri_connectivity[t_cnt] = m_tri_connectivity[i];
            m_tri_connectivity[t_cnt].hash = 0;
            if (p_face_attrs) p_face_attrs->move(i, t_cnt);

            for (auto j = 0; j < 3; j++) {
                if (p_edge_attrs) p_edge_attrs->move(i * 3 + j, t_cnt * 3 + j);
            }
        }
        for (size_t& v_id : m_tri_connectivity[t_cnt].m_indices) v_id = map_v_ids[v_id];
        t_cnt++;
    }

    current_vert_size = v_cnt;
    current_tri_size = t_cnt;

    m_vertex_connectivity.m_attributes.resize(v_cnt);
    m_vertex_connectivity.shrink_to_fit();
    m_tri_connectivity.m_attributes.resize(t_cnt);
    m_tri_connectivity.shrink_to_fit();

    resize_mutex(vert_capacity());

    // Resize user class attributes
    if (p_vertex_attrs) p_vertex_attrs->resize(vert_capacity());
    if (p_edge_attrs) p_edge_attrs->resize(tri_capacity() * 3);
    if (p_face_attrs) p_face_attrs->resize(tri_capacity());


    // update constraints(edge tuple pairs)
    for (int i = 0; i < tri_capacity(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            auto cur_t = tuple_from_edge(i, j);
            if (is_boundary_edge(cur_t))
            {
                auto pair_t = edge_attrs[3 * i + j].pair;
                edge_attrs[3 * i + j].pair = tuple_from_edge(map_t_ids[pair_t.eid_unsafe(*this) / 3], pair_t.eid_unsafe(*this) % 3);
            } 
        }
    }
}

bool check_constraints(double eps = 1e-7)
{
    auto all_edges = this->get_edges();
    bool flag = true;
    for (auto t_e : all_edges)
    {
        if (!this->is_boundary_edge(t_e)) continue;
        auto t_e_pair = edge_attrs[t_e.eid(*this)].pair;
        int v0 = t_e.vid(*this);
        int v1 = t_e.switch_vertex(*this).vid(*this);
        int v2 = t_e_pair.vid(*this);
        int v3 = t_e_pair.switch_vertex(*this).vid(*this);
        
        // check length
        auto e_ab = (vertex_attrs[v1].pos - vertex_attrs[v0].pos);
        auto e_dc = (vertex_attrs[v2].pos - vertex_attrs[v3].pos);
        // std::cout << "(" << v0 << "," << v1 << ") - (" << v2 << "," << v3 << ")" << std::endl;
        // std::cout << abs(e_ab.norm() - e_dc.norm()) << std::endl;
        if (abs(e_ab.norm() - e_dc.norm()) > eps) 
        {
            std::cout << "length error " << "(" << v0 << "," << v1 << ") - (" << v2 << "," << v3 << ")" << std::endl;
            std::cout << abs(e_ab.norm() - e_dc.norm()) << std::endl;
            flag = false;
            // return false;
        } 
    
        // check angle
        Eigen::Vector2d e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        double index = 2 * angle / igl::PI;
        // std::cout << index << std::endl;
        if (abs(index - round(index)) > eps)
        {
            std::cout << "angle error " << "(" << v0 << "," << v1 << ") - (" << v2 << "," << v3 << ")" << std::endl;
            std::cout << index << std::endl;
            flag = false;
            // return false;
        }
    }

    return flag;
    // return true;
}

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

// Exports V and F of the stored mesh
void export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv);

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

// Edge Splitting
bool split_edge_before(const Tuple& t) override;
bool split_edge_after(const Tuple& t) override;
void split_all_edges();

};

} // namespace tetwild

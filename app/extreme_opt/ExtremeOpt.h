#pragma once

#include <igl/Timer.h>
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
        if (before(t, m))
            {
                auto new_t = m.collapse_edge_new(t, new_tris);
                // m.collapse_edge_new(m.edge_attrs[t.eid(m)].pair, new_tris);
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

struct PositionInfoCache
{
    Eigen::Vector3d V1;
    Eigen::Vector3d V2;
    Eigen::Vector2d uv1;
    Eigen::Vector2d uv2;
    double E_max_before_collpase;
};
tbb::enumerable_thread_specific<PositionInfoCache> position_cache;

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

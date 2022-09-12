#pragma once

#include <igl/Timer.h>
#include <wmtk/TriMesh.h>


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
};

class ExtremeOpt : public wmtk::TriMesh
{
public:

// Energy Assigned to undefined energy
// TODO: why not the max double?
const double MAX_ENERGY = 1e50;

ExtremeOpt() {};

virtual ~ExtremeOpt() {};


// Store the per-vertex attributes
wmtk::AttributeCollection<VertexAttributes> vertex_attrs;

// Initializes the mesh
void create_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv);

// Exports V and F of the stored mesh
void export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

// Writes a triangle mesh in OBJ format
void write_obj(const std::string& path);

// Computes the quality of a triangle
double get_quality(const Tuple& loc) const;

// Computes the average quality of a mesh
Eigen::VectorXd get_quality_all_triangles();

// Check if a triangle is inverted
bool is_inverted(const Tuple& loc) const;

// Smoothing
void smooth_all_vertices();
bool smooth_before(const Tuple& t) override;
bool smooth_after(const Tuple& t) override;


};

} // namespace tetwild

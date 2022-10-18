#include "ExtremeOpt.h"
#include <Eigen/Core>
#include <igl/write_triangle_mesh.h>
#include <wmtk/utils/AMIPS2D.h>
#include <igl/predicates/predicates.h>
#include <tbb/concurrent_vector.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>

namespace extremeopt {

void ExtremeOpt::create_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv)
{
    // Register attributes
    p_vertex_attrs = &vertex_attrs;
    p_face_attrs = &face_attrs;

    Eigen::VectorXd dblarea;
    igl::doublearea(V, F, dblarea);
    
    // Convert from eigen to internal representation (TODO: move to utils and remove it from all app)
    std::vector<std::array<size_t, 3>> tri(F.rows());
    
    for (int i = 0; i < F.rows(); i++)
    {
        for (int j = 0; j < 3; j++) 
            tri[i][j] = (size_t)F(i, j);
    }
    
    // Initialize the trimesh class which handles connectivity
    wmtk::TriMesh::create_mesh(V.rows(), tri);
    
    // Save the face area in the face attributes
    for (int i = 0; i < F.rows(); i++)
    {
        face_attrs[i].area_3d = dblarea[i];
    }
    // Save the vertex position in the vertex attributes
    for (unsigned i = 0; i<V.rows();++i)
    {
        vertex_attrs[i].pos << uv.row(i)[0], uv.row(i)[1];
        vertex_attrs[i].pos_3d << V.row(i)[0], V.row(i)[1], V.row(i)[2];
    }
    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);
    for (auto bd : bds)
    {
        for (int vec : bd)
        {
            vertex_attrs[vec].fixed = true;
        }
    }
}

void ExtremeOpt::export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv)
{   
    consolidate_mesh();
    V = Eigen::MatrixXd::Zero(vert_capacity(), 3);
    uv = Eigen::MatrixXd::Zero(vert_capacity(), 2);
    for (auto& t : get_vertices()) {
        auto i = t.vid(*this);
        V.row(i) = vertex_attrs[i].pos_3d;
        uv.row(i) = vertex_attrs[i].pos;
    }

    F = Eigen::MatrixXi::Constant(tri_capacity(), 3, -1);
    for (auto& t : get_faces()) {
        auto i = t.fid(*this);
        auto vs = oriented_tri_vertices(t);
        for (int j = 0; j < 3; j++) {
            F(i, j) = vs[j].vid(*this);
        }
    }
}

void ExtremeOpt::write_obj(const std::string& path)
{
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;

    export_mesh(V,F,uv);

    igl::writeOBJ(path,V,F,V,F,uv,F);
}


double ExtremeOpt::get_quality(const Tuple& loc) const
{
    // Global ids of the vertices of the triangle
    auto its = oriented_tri_vids(loc);

    // Temporary variable to store the stacked coordinates of the triangle
    std::array<double, 6> T;
    auto energy = -1.;
    for (auto k = 0; k < 3; k++)
        for (auto j = 0; j < 2; j++) 
            T[k * 2 + j] = vertex_attrs[its[k]].pos[j];

    // Energy evaluation
    energy = wmtk::AMIPS2D_energy(T);

    // Filter for numerical issues
    if (std::isinf(energy) || std::isnan(energy)) 
        return MAX_ENERGY;

    return energy;
}

Eigen::VectorXd ExtremeOpt::get_quality_all_triangles()
{
    // Use a concurrent vector as for_each_face is parallel
    tbb::concurrent_vector<double> quality;
    quality.reserve(vertex_attrs.size());

    // Evaluate quality in parallel
    for_each_face(
        [&](auto& f) {
            quality.push_back(get_quality(f));
        }
    );

    // Copy back in a VectorXd
    Eigen::VectorXd ret(quality.size());
    for (unsigned i=0; i<quality.size();++i)
        ret[i] = quality[i];
    return ret;
}

void ExtremeOpt::get_mesh_onering(const Tuple& t, Eigen::MatrixXd &V_local, Eigen::MatrixXd &uv_local, Eigen::MatrixXi &F_local)
{
    auto vid = t.vid(*this);
    auto vid_onering = get_one_ring_vids_for_vertex(vid);
    auto locs = get_one_ring_tris_for_vertex(t);
    V_local.resize(vid_onering.size(), 3);
    uv_local.resize(vid_onering.size(), 2);
    for (size_t i = 0; i < vid_onering.size(); i++) {
        V_local.row(i) = vertex_attrs[vid_onering[i]].pos_3d;
        uv_local.row(i) = vertex_attrs[vid_onering[i]].pos;
    }
    std::vector<int> v_map(vertex_attrs.size(), -1);
    for (size_t i = 0; i < vid_onering.size(); i++) {
        v_map[vid_onering[i]] = i;
    }
    F_local.resize(locs.size(), 3);
    for (size_t i = 0; i < locs.size(); i++) {
        int t_id = locs[i].fid(*this);
        auto local_tuples = oriented_tri_vertices(locs[i]);
        for (size_t j = 0; j < 3; j++) {
            F_local(i, j) = v_map[local_tuples[j].vid(*this)];
        }
    }
}

bool ExtremeOpt::is_inverted(const Tuple& loc) const
{
    // Get the vertices ids
    auto vs = oriented_tri_vertices(loc);

    igl::predicates::exactinit();

    // Use igl for checking orientation
    auto res = igl::predicates::orient2d(
        vertex_attrs[vs[0].vid(*this)].pos,
        vertex_attrs[vs[1].vid(*this)].pos,
        vertex_attrs[vs[2].vid(*this)].pos);

    // The element is inverted if it not positive (i.e. it is negative or it is degenerate)
    return (res != igl::predicates::Orientation::POSITIVE);
}

}
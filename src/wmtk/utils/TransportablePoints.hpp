#pragma once


#include <wmtk/TriMesh.h>


namespace wmtk {
class TransportablePointsBase
{
    ~TransportablePointsBase();
    // convenience function that just calls
    void before_hook(const TriMesh& m, const std::set<size_t>& input_tris);
    void after_hook(const TriMesh& m, const std::set<size_t>& output_tris);

    // derived class is required to store a global representation of the point, used in before_hook
    virtual void apply_barycentric(const TriMesh& m, size_t point_index) = 0;

    // derived class is required to identify which point and triangle
    virtual bool point_in_triangle(const TriMesh& m, size_t triangle_index, size_t point_index)
        const = 0;

    virtual std::array<double, 3>
    get_barycentric(const TriMesh& m, size_t triangle_index, size_t point_index) const = 0;


    void update_local_coordinates(
        const TriMesh& m,
        const std::set<size_t>& input_tris,
        const std::set<size_t>& output_tris);

protected:
    // local representation of points in a triangle mesh
    tbb::concurrent_vector<size_t> triangle_indices;
    tbb::concurrent_vector<std::array<double, 3>> barycentric_coordinates;

    tbb::enumerable_thread_specific<std::set<size_t>> active_points;
};

template <typename PointType>
class TransportablePoints
{
    // derived class is required to store a global representation of the point, used in before_hook
    void apply_barycentric(const TriMesh& m, size_t point_index) override;

    // derived class is required to identify which point and triangle
    bool point_in_triangle(const TriMesh& m, size_t triangle_index, size_t point_index)
        const override;

    std::array<double, 3>
    get_barycentric(const TriMesh& m, size_t triangle_index, size_t point_index) const override;


    // global coordinates
    tbb::concurrent_vector<PointType> points_global;
};


template <typename PointType>
void TransportablePoints<PointType>::global_from_local(const TriMesh& m)
{}
template <typename PointType>
void TransportablePoints<PointType>::local_from_global(const TriMesh& m)
{}

} // namespace wmtk

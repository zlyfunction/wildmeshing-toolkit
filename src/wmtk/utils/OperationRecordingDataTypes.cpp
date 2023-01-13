#include <wmtk/utils/OperationRecordingDataTypes.hpp>

using namespace wmtk;

HighFive::CompoundType AttributeChanges::datatype()
{
    return HighFive::CompoundType{
        {"attribute_name", HighFive::create_datatype<char[20]>()},
        {"attribute_size", HighFive::create_datatype<size_t>()},
        {"change_range_begin", HighFive::create_datatype<size_t>()},
        {"change_range_end", HighFive::create_datatype<size_t>()}};
}

HighFive::CompoundType TriMeshOperationData::datatype()
{
    return HighFive::CompoundType{
        {"name", HighFive::create_datatype<char[20]>()},
        {"triangle_id", HighFive::create_datatype<size_t>()},
        {"local_edge_id", HighFive::create_datatype<size_t>()},
        {"vertex_id", HighFive::create_datatype<size_t>()},
        {"update_range_begin", HighFive::create_datatype<size_t>()},
        {"update_range_end", HighFive::create_datatype<size_t>()}};
}

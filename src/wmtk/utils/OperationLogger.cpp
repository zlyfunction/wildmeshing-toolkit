#include <wmtk/TriMesh.h>
#include <wmtk/utils/AttributeRecorder.h>
#include <wmtk/utils/OperationLogger.h>
#include <wmtk/utils/Logger.hpp>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5File.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#include <ostream>
#include <wmtk/utils/Logger.hpp>
#include <wmtk/utils/OperationRecordingDataTypes.hpp>

HIGHFIVE_REGISTER_TYPE(wmtk::AttributeChanges, wmtk::AttributeChanges::datatype);
HIGHFIVE_REGISTER_TYPE(wmtk::TriMeshOperation, wmtk::TriMeshOperation::datatype);
namespace wmtk {

namespace {

template <typename T>
HighFive::DataSet create_dataset(HighFive::File& file, const std::string& name)
{
    if (file.exist(name)) {
        if (HighFive::ObjectType::Dataset == file.getObjectType(name)) {
            return file.getDataSet(name);
        } else {
            logger().error(
                "create_dataset: {} had root node {} but it was not a dataset",
                file.getName(),
                name);
        }
    }
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking(std::vector<hsize_t>{2}));

    return file.createDataSet(
        std::string(name),
        // create an empty dataspace of unlimited size
        HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
        // configure its datatype according to derived class's datatype spec
        HighFive::create_datatype<T>(),
        // should enable chunking to allow appending
        props);
}

} // namespace

OperationLogger::OperationLogger(HighFive::File& f)
    : file(f)
    , operation_dataset(create_dataset<TriMeshOperation>(f, "operations"))
    , attribute_changes_dataset(create_dataset<AttributeChanges>(f, "attribute_changes"))
{}
OperationLogger::~OperationLogger() = default;


template <size_t Size>
OperationRecorder::OperationRecorder(
    OperationLogger& logger_,
    OperationType type,
    const std::string_view& cmd,
    const std::array<size_t, Size>& tuple)
    : OperationRecorder(logger_, type, cmd, tuple.data(), Size)
{}
auto OperationLogger::start(

    const TriMesh& m,
    const std::string_view& cmd,
    const std::array<size_t, 3>& tuple) -> OperationRecorder
{
    return OperationRecorder(*this, OperationRecorder::OperationType::TriMesh, cmd, tuple);
}

auto OperationLogger::start_ptr(
    const TriMesh& m,
    const std::string_view& cmd,
    const std::array<size_t, 3>& tuple) -> OperationRecorder::Ptr
{
    return std::make_shared<OperationRecorder>(
        *this,
        OperationRecorder::OperationType::TriMesh,
        cmd,
        tuple);
}


OperationRecorder::OperationRecorder(
    OperationLogger& logger_,
    OperationType type_,
    const std::string_view& cmd,
    const size_t* tuple,
    size_t tuple_size)
    : logger(logger_)
    , type(type_)
    , name(cmd)
    , tuple_data(tuple, tuple + tuple_size)
{
    // nlohmann::json& js = *data;
    // js["operation"] = cmd;
    // auto& tup = js["tuple"];
    // for (size_t j = 0; j < tuple_size; ++j) {
    //     tup.push_back(tuple[j]);
    // }
}

void OperationRecorder::OperationRecorder::cancel()
{
    cancelled = true;
}

OperationRecorder::~OperationRecorder()
{ // only lock the mutex when we output to the output stream
    if (!cancelled) {
        tbb::mutex::scoped_lock lock(logger.output_mutex);


        // commit update attributes
        std::vector<AttributeChanges> changes;

        for (auto&& [attr_name, p_recorder] : logger.attribute_recorders) {
            auto [start, end] = p_recorder->record();
            changes.emplace_back(AttributeChanges{attr_name, start, end});
        }
        auto [start, end] = append_values_to_1d_dataset(logger.attribute_changes_dataset, changes);

        // commit command itself
        switch (this->type) {
        case OperationType::TriMesh: {
            TriMeshOperation op;
            strncpy(
                op.name,
                name.c_str(),
                sizeof(name) /
                    sizeof(char)); // yes sizeof(char)==1, maybe chartype changes someday?
            assert(tuple_data.size() == 3);
            op.triangle_id = tuple_data[0];
            op.local_edge_id = tuple_data[1];
            op.vertex_id = tuple_data[2];
            op.update_range_begin = start;
            op.update_range_end = end;


            auto size = append_value_to_1d_dataset(logger.operation_dataset, op);
        } break;
        case OperationType::TetMesh: {
        } break;
        }
    } else {
    }
}

void OperationLogger::add_attribute_recorder(
    std::string&& name,
    AttributeCollectionRecorderBase& attribute_recorder)
{
    attribute_recorders.emplace(std::move(name), &attribute_recorder);
}

/*
void OperationLogger::record_attributes()
{
    for (auto&& [name, attr_recorder] : attribute_recorders) {
        attr_recorder->record(dataset, name);
    }
}
bool OperationLogger::record_attribute(const std::string& attribute_name)
{
    if (auto it = attribute_recorders.find(attribute_name); it != attribute_recorders.end()) {
        attr_recorder->record(dataset, name);
    } else {
        return false;
    }
    return true;
}
*/
} // namespace wmtk


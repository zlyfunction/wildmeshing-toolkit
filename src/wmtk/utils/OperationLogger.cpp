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

using namespace wmtk;
namespace {

// checks whether the file holding the dataset exists
bool does_dataset_exist(const HighFive::File& file, const std::string& name)
{
    auto obj_names = file.listObjectNames();
    spdlog::warn("Object names when looking for {}:  {}", name, fmt::join(obj_names, ","));
    if (file.exist(name)) {
        return true;
        ;
    }
    // for whaterver reason file.exist doesn't work with my invocation so doing it the dumb way
    for (auto&& n : obj_names) {
        if (n == name) {
            if (HighFive::ObjectType::Dataset == file.getObjectType(name)) {
                return true;
            } else {
                logger().error(
                    "create_dataset: {} had root node {} but it was not a dataset",
                    file.getName(),
                    name);
            }
        }
    }
    return false;
}
HighFive::DataSet
create_dataset(HighFive::File& file, const std::string& name, const HighFive::DataType& datatype)
{
    spdlog::info("Creating dataset {}", name);

    if (does_dataset_exist(file, name)) {
        auto ds = file.getDataSet(name);
        spdlog::info("Returning dataset {} with {} entries", name, ds.getElementCount());
        return ds;
    } else {
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{2}));
        return file.createDataSet(
            std::string(name),
            // create an empty dataspace of unlimited size
            HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
            // configure its datatype according to derived class's datatype spec
            datatype,
            // should enable chunking to allow appending
            props);
    }
}
template <typename T>
HighFive::DataSet create_dataset(HighFive::File& file, const std::string& name)
{
    return create_dataset(file, name, HighFive::create_datatype<T>());
}

} // namespace

void OperationLogger::set_readonly()
{
    read_mode = true;
}
bool OperationLogger::is_readonly()
{
    return read_mode;
}
OperationLogger::OperationLogger(HighFive::File& f, const HighFive::DataType& op_type)
    : file(f)
    , operation_dataset(::create_dataset(f, "operations", op_type))
    , attribute_changes_dataset(::create_dataset<AttributeChanges>(f, "attribute_changes"))
{}
OperationLogger::~OperationLogger() = default;

HighFive::DataSet OperationLogger::create_dataset(
    const std::string& name,
    const HighFive::DataType& datatype)
{
    return ::create_dataset(file, name, datatype);
}


OperationRecorder::OperationRecorder(OperationLogger& logger_, const std::string& cmd)
    : logger(logger_)
    , name(cmd)
{}
void OperationRecorder::OperationRecorder::cancel()
{
    is_cancelled = true;
}

OperationRecorder::~OperationRecorder()
{
    if (!cancelled()) {
        spdlog::error(
            "Recorder cannot be destroyed without being flushed! derived class must flush!");
    }
}

void OperationRecorder::flush()
{
    // only lock the mutex when we output to the output stream
    if (!cancelled()) {
        tbb::mutex::scoped_lock lock(logger.output_mutex);


        // commit update attributes
        std::vector<AttributeChanges> changes;

        for (auto&& [attr_name, p_recorder] : logger.attribute_recorders) {
            auto [start, end, size] = p_recorder->record();
            changes.emplace_back(AttributeChanges{attr_name, size, start, end});
        }
        auto [start, end] = append_values_to_1d_dataset(logger.attribute_changes_dataset, changes);

        // commit command itself
        commit(start, end);
    }
    is_cancelled = true;
}
size_t OperationLogger::attribute_changes_count() const
{
    return attribute_changes_dataset.getElementCount();
}

size_t OperationLogger::operation_count() const
{
    return operation_dataset.getElementCount();
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


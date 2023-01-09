#pragma once

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/mutex.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
#include <highfive/H5DataSet.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#include <memory>
#include <nlohmann/json.hpp>
#include <string_view>
#include <wmtk/utils/Logger.hpp>

namespace HighFive {
class File;
}

namespace wmtk {
class TriMesh;
class OperationLogger;
class AttributeCollectionRecorderBase;
;


class OperationRecorder
{
public:
    enum class OperationType { TriMesh, TetMesh };
    // struct UpdateData;
    struct OperationData;
    using Ptr = std::shared_ptr<OperationRecorder>;


    template <size_t Size>
    OperationRecorder(
        OperationLogger& logger,
        OperationType type,
        const std::string_view& cmd,
        const std::array<size_t, Size>& tuple);


    // grr std::span would be nice here
    OperationRecorder(
        OperationLogger& logger,
        OperationType type,
        const std::string_view& cmd,
        const size_t* tuple,
        size_t tuple_size);
    ~OperationRecorder();

    // If the operation
    void cancel();


private:
    OperationLogger& logger;
    OperationType type;
    std::string name;
    std::vector<size_t> tuple_data; // Tet/Tri use different sizes
    std::vector<std::pair<std::string, std::array<size_t, 2>>> attribute_updates;
    bool cancelled = false;
};

class OperationLogger
{
public:
    friend class OperationRecorder;
    friend class OperationReplayer;
    OperationLogger(HighFive::File& file);
    ~OperationLogger();
    OperationRecorder
    start(const TriMesh& m, const std::string_view& cmd, const std::array<size_t, 3>& tuple);
    OperationRecorder::Ptr
    start_ptr(const TriMesh& m, const std::string_view& cmd, const std::array<size_t, 3>& tuple);
    void add_attribute_recorder(
        std::string&& attribute_name,
        AttributeCollectionRecorderBase& attribute_recorder);
    size_t operation_count() const;
    size_t attribute_changes_count() const;


private:
    oneapi::tbb::mutex output_mutex;
    HighFive::File& file;
    HighFive::DataSet operation_dataset;
    HighFive::DataSet attribute_changes_dataset;
    // std::ostream& output_stream;
    std::map<std::string, AttributeCollectionRecorderBase*> attribute_recorders;


    // returns true if attribute was successfully recorded
    std::array<size_t, 2> record_attribute(const std::string& attribute_name);
};


} // namespace wmtk

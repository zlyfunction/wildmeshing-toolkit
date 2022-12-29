#pragma once

#include <oneapi/tbb/mutex.h>
#include <wmtk/TriMesh.h>
#include <memory>
#include <string_view>


namespace wmtk {

class OperationLogger;
class OperationRecorder
{
public:
    struct Data;
    using Ptr = std::shared_ptr<OperationRecorder>;

    OperationRecorder(
        OperationLogger& logger,
        const TriMesh& m,
        const std::string_view& cmd,
        const TriMesh::Tuple& tuple);
    ~OperationRecorder();

    // If the operation
    void cancel();

private:
    OperationLogger& logger;
    std::unique_ptr<Data> data;
};

class OperationLogger
{
public:
    friend class OperationRecorder;
    OperationLogger(std::ostream& output_stream);
    ~OperationLogger();
    OperationRecorder
    start(const TriMesh& m, const std::string_view& cmd, const TriMesh::Tuple& tuple);
    OperationRecorder::Ptr
    start_ptr(const TriMesh& m, const std::string_view& cmd, const TriMesh::Tuple& tuple);


private:
    oneapi::tbb::mutex output_mutex;
    std::ostream& output_stream;
};

} // namespace wmtk

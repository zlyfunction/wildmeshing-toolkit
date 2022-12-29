#include <wmtk/TriMesh.h>
#include <wmtk/utils/OperationLogger.h>
#include <nlohmann/json.hpp>
#include <ostream>
#include <wmtk/utils/Logger.hpp>

namespace wmtk {
OperationLogger::OperationLogger(std::ostream& os)
    : output_stream(os)
{}
OperationLogger::~OperationLogger() = default;


auto OperationLogger::start(

    const TriMesh& m,
    const std::string_view& cmd,
    const TriMesh::Tuple& tuple) -> OperationRecorder
{
    return OperationRecorder(*this, m, cmd, tuple);
}

auto OperationLogger::start_ptr(
    const TriMesh& m,
    const std::string_view& cmd,
    const TriMesh::Tuple& tuple) -> OperationRecorder::Ptr
{
    return std::make_shared<OperationRecorder>(*this, m, cmd, tuple);
}

struct OperationRecorder::OperationRecorder::Data : public nlohmann::json
{
};

OperationRecorder::OperationRecorder(
    OperationLogger& logger_,
    const TriMesh& m,
    const std::string_view& cmd,
    const TriMesh::Tuple& tuple)
    : logger(logger_)
    , data(std::make_unique<Data>())
{
    nlohmann::json& js = *data;
    js["operation"] = cmd;
    js["tuple"] = {tuple.vid(m), tuple.local_eid(m), tuple.fid(m)};
}

void OperationRecorder::OperationRecorder::cancel()
{
    data.reset();
}

OperationRecorder::~OperationRecorder()
{ // only lock the mutex when we output to the output stream
    if (data) {
        tbb::mutex::scoped_lock lock(logger.output_mutex);
        logger.output_stream << *data << '\n';
    }
}
} // namespace wmtk

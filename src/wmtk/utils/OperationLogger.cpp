#include <wmtk/utils/OperationLogger.h>
#include <wmtk/TriMesh.h>
#include <nlohmann/json.hpp>
#include <ostream>
#include <wmtk/utils/Logger.hpp>

namespace wmtk {
OperationLogger::OperationLogger(std::ostream& os)
    : output_stream(os)
{}

void OperationLogger::log(
    const TriMesh& m,
    const std::string_view& cmd,
    const TriMesh::Tuple& tuple)
{
    nlohmann::json js;
    js["operation"] = cmd;
    js["tuple"] = {tuple.vid(m), tuple.local_eid(m), tuple.fid(m)};

    { // only lock the mutex when we output to the output stream
        tbb::mutex::scoped_lock lock(output_mutex);
        output_stream << js << '\n';
    }
}
} // namespace wmtk

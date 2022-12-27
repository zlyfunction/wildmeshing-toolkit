#pragma once

#include <oneapi/tbb/mutex.h>
#include <wmtk/TriMesh.h>
#include <string_view>


namespace wmtk {

class OperationLogger
{
public:
    OperationLogger(std::ostream& output_stream);
    void log(const TriMesh& m, const std::string_view& cmd, const TriMesh::Tuple& tuple);

private:
    oneapi::tbb::mutex output_mutex;
    std::ostream& output_stream;
};
} // namespace wmtk

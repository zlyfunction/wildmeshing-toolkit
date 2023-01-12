#pragma once
#include <wmtk/TriMesh.h>

namespace wmtk {
class TriMeshOperation
{
public:
    using Tuple = TriMesh::Tuple;
    struct ExecuteReturnData
    {
        Tuple tuple;
        std::vector<Tuple> new_tris;
        bool success = false;
    };

    ExecuteReturnData operator()(const Tuple& t, TriMesh& m);
    virtual std::string name() const = 0;


    TriMeshOperation() {}
    virtual ~TriMeshOperation() {}

protected:
    // returns the changed tris + whether success occured
    virtual ExecuteReturnData execute(const Tuple& t, TriMesh& m) = 0;
    virtual bool before_check(const Tuple& t, TriMesh& m) = 0;
    virtual bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) = 0;
    virtual bool invariants(const ExecuteReturnData& ret_data, TriMesh& m);


#if defined(USE_OPERATION_LOGGER)
    std::weak_ptr<OperationRecorder> recorder(TriMesh& m) const;
#endif

    void set_vertex_size(size_t size, TriMesh& m);
    void set_tri_size(size_t size, TriMesh& m);
};

class TriMeshSplitEdgeOperation : public TriMeshOperation
{
public:
    ExecuteReturnData execute(const Tuple& t, TriMesh& m) override;
    bool before_check(const Tuple& t, TriMesh& m) override;
    bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) override;
    std::string name() const override;
};
class TriMeshSwapEdgeOperation : public TriMeshOperation
{
public:
    ExecuteReturnData execute(const Tuple& t, TriMesh& m) override;
    bool before_check(const Tuple& t, TriMesh& m) override;
    bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) override;
    std::string name() const override;
};

class TriMeshEdgeCollapseOperation : public TriMeshOperation
{
public:
    ExecuteReturnData execute(const Tuple& t, TriMesh& m) override;
    bool before_check(const Tuple& t, TriMesh& m) override;
    bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) override;
    std::string name() const override;
};

class TriMeshSmoothVertexOperation : public TriMeshOperation
{
public:
    ExecuteReturnData execute(const Tuple& t, TriMesh& m) override;
    bool before_check(const Tuple& t, TriMesh& m) override;
    bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) override;
    std::string name() const override;
    bool invariants(const ExecuteReturnData& ret_data, TriMesh& m) override;
};

class TriMeshConsolidateOperation : public TriMeshOperation
{
public:
    ExecuteReturnData execute(const Tuple& t, TriMesh& m) override;
    bool before_check(const Tuple& t, TriMesh& m) override;
    bool after_check(const ExecuteReturnData& ret_data, TriMesh& m) override;
    std::string name() const override;
};
} // namespace wmtk

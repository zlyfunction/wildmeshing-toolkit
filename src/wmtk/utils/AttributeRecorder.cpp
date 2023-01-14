#include <wmtk/utils/AttributeRecorder.h>
#include <wmtk/AttributeCollection.hpp>
#include <wmtk/utils/Hdf5Utils.h>

using namespace wmtk;
AttributeCollectionRecorderBase::AttributeCollectionRecorderBase(
    HighFive::File& file,
    const std::string& name,
    const HighFive::DataType& data_type)
    : AttributeCollectionRecorderBase(
            create_dataset(file, name, data_type))
{}

AttributeCollectionRecorderBase::AttributeCollectionRecorderBase(HighFive::DataSet&& dataset_)
    : dataset(dataset_)
{}

AttributeCollectionRecorderBase::~AttributeCollectionRecorderBase() = default;

HighFive::DataSetAccessProps AttributeCollectionRecorderBase::access_properties()
{
    HighFive::DataSetAccessProps props;
    return props;
}
HighFive::DataSetCreateProps AttributeCollectionRecorderBase::create_properties()
{
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking(std::vector<hsize_t>{2}));
    return props;
}

std::array<size_t, 3> AttributeCollectionRecorderBase::record()
{
    return record(dataset);
}

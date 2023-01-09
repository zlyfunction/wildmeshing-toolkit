#include <wmtk/AttributeCollection.hpp>
#include <wmtk/utils/AttributeRecorder.h>

WMTK_HDF5_REGISTER_ATTRIBUTE_TYPE(double)
using namespace wmtk;
AttributeCollectionRecorderBase::AttributeCollectionRecorderBase(
    HighFive::File& file,
    const std::string_view& name, const HighFive::DataType& data_type)
    : AttributeCollectionRecorderBase(file.createDataSet(
          std::string(name),
          // create an empty dataspace of unlimited size
          HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
          // configure its datatype according to derived class's datatype spec
          data_type,
          // should enable chunking to allow appending
          create_properties(),
          access_properties()))
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

std::array<size_t, 2> AttributeCollectionRecorderBase::record()
{
    return record(dataset);
}

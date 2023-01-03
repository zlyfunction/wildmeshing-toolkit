#pragma once
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5File.hpp>
#include <wmtk/AttributeCollection.hpp>
#include <wmtk/utils/OperatorRecordingDataTypes.hpp>


namespace wmtk {


// Interface for adding an attribute to a logged hdf5 file
class AttributeCollectionRecorderBase
{
public:
    AttributeCollectionRecorderBase(
        HighFive::File& file,
        const std::string_view& name,
        const HighFive::DataType& data_type);
    AttributeCollectionRecorderBase(HighFive::DataSet&& dataset_);
    virtual ~AttributeCollectionRecorderBase();
    static HighFive::CompoundType record_datatype();
    static HighFive::DataSetCreateProps create_properties();
    static HighFive::DataSetAccessProps access_properties();
    // returns the range of values used
    virtual std::array<size_t, 2> record(HighFive::DataSet& data_set) = 0;

    // returns the range of values used
    std::array<size_t, 2> record();

private:
    HighFive::DataSet dataset;
};


// Class capable of recording updates to a single attribute
template <typename T>
class AttributeCollectionRecorder : public AttributeCollectionRecorderBase
{
public:
    struct UpdateData;
    static HighFive::CompoundType datatype();
    AttributeCollectionRecorder(
        HighFive::File& file,
        const std::string_view& name,
        AttributeCollection<T>& attr_);


    std::array<size_t, 2> record(HighFive::DataSet& data_set) override;
    using AttributeCollectionRecorderBase::record;

private:
    AttributeCollection<T>& attribute_collection;
};

// deduction guide
template <typename T>
AttributeCollectionRecorder(
    HighFive::File& file,
    const std::string_view& name,
    const AttributeCollection<T>&) -> AttributeCollectionRecorder<T>;

template <typename T>
AttributeCollectionRecorder<T>::AttributeCollectionRecorder(
    HighFive::File& file,
    const std::string_view& name,
    AttributeCollection<T>& attr_)
    : AttributeCollectionRecorderBase(file, name, datatype())
    , attribute_collection(attr_)
{}

template <typename T>
HighFive::CompoundType AttributeCollectionRecorder<T>::datatype()
{
    return HighFive::CompoundType{
        {"index", HighFive::create_datatype<size_t>()},
        {"old_value", HighFive::create_datatype<T>()},
        {"new_value", HighFive::create_datatype<T>()}};
}


template <typename T>
std::array<size_t, 2> AttributeCollectionRecorder<T>::record(HighFive::DataSet& data_set)
{
    const std::map<size_t, T>& rollback_list = attribute_collection.m_rollback_list.local();
    const tbb::concurrent_vector<T>& attributes = attribute_collection.m_attributes;

    std::vector<UpdateData> data;
    data.reserve(rollback_list.size());
    std::transform(
        rollback_list.begin(),
        rollback_list.end(),
        std::back_inserter(data),
        [&attributes](const std::pair<const size_t, T>& pr) -> UpdateData {
            const auto& [index, new_value] = pr;
            const T& old_value = attributes[index];
            return UpdateData{index, old_value, new_value};
        });

    return append_values_to_1d_dataset(data_set, data);
}


} // namespace wmtk


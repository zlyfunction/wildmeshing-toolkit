
#include <wmtk/utils/OperationLogger.h>
#include <wmtk/utils/AttributeRecorder.h>
#include <wmtk/AttributeCollection.hpp>

#include <catch2/catch.hpp>
#include <iostream>

using namespace wmtk;


//WMTK_HDF5_REGISTER_ATTRIBUTE_TYPE(double)

TEST_CASE("create_recorder", "[attribute_recording]")
{
    /*
    using namespace HighFive;
    File file("create_recorder.hd5", File::ReadWrite | File::Create | File::Truncate);

    size_t size = 20;

    AttributeCollection<double> attribute_collection;
    attribute_collection.resize(size);
    AttributeCollectionRecorder<double> attribute_recorder(file, "attribute1", attribute_collection);

    for(size_t j = 0; j < attribute_collection.size(); ++j) {
        attribute_collection[j] = -double(j);
    }

    attribute_collection.begin_protect();


    for(size_t j = 0; j < attribute_collection.size(); ++j) {
        attribute_collection[j] = double(2 * j);
    }

    for(size_t j = 0; j < attribute_collection.size(); ++j) {
        REQUIRE(attribute_collection[j] == 2 * j);
    }

    const auto& rollback_list = attribute_collection.m_rollback_list.local();


    REQUIRE(rollback_list.size() <= attribute_collection.size());

    {
        auto [start,end] = attribute_recorder.record();
        CHECK(start==0);
        CHECK(end==20);
    }

    attribute_collection.end_protect();

    attribute_collection.begin_protect();
    for(size_t j = 0; j < attribute_collection.size(); j+=2) {
        attribute_collection[j] = double(3 * j);
    }
    {
        auto [start,finish] = attribute_recorder.record();
        CHECK(start == size);
        CHECK(finish == (size + size/2));
    }
    attribute_collection.end_protect();

    for(size_t j = 0; j < attribute_collection.size(); ++j) {
        if(j%2 == 0) {
        REQUIRE(attribute_collection[j] == 3 * j);
        } else {
        REQUIRE(attribute_collection[j] == 2 * j);
        }
    }

    */

}


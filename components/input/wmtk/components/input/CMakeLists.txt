set(COMPONENT_NAME input)
add_component(${COMPONENT_NAME})
if(NOT ${WMTK_ENABLE_COMPONENT_${COMPONENT_NAME}})
    return()
endif()
set(SRC_FILES
    internal/InputOptions.hpp
    internal/mesh_with_tag_from_image.cpp
    internal/mesh_with_tag_from_image.hpp
    input.cpp
	input.hpp)

include(stb)



target_sources(wmtk_${COMPONENT_NAME} PRIVATE ${SRC_FILES})
target_link_libraries(wmtk_${COMPONENT_NAME} PRIVATE stb::image)

add_component_test(wmtk::${COMPONENT_NAME} tests/input.cpp)

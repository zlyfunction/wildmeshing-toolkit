set(COMPONENT_NAME tet_remeshing)
add_component(${COMPONENT_NAME})
if(NOT ${WMTK_ENABLE_COMPONENT_${COMPONENT_NAME}})
    return()
endif()
set(SRC_FILES
    tet_remeshing.hpp
    tet_remeshing.cpp
    TetRemeshingOptions.hpp
    )
target_sources(wmtk_${COMPONENT_NAME} PRIVATE ${SRC_FILES})
# target_link_libraries(wmtk_${COMPONENT_NAME} PRIVATE
#     #
# )

function(wmtk_link_components TARGET_NAME ...)

    list(REMOVE_AT ARGV 0)
    foreach(COMPONENT_NAME ${ARGV})
    if(NOT WMTK_ENABLE_COMPONENT_${COMPONENT_NAME})
        message(FATAL_ERROR "Cannot build ${TARGET_NAME} without component wmtk::${COMPONENT_NAME}. Set WMTK_ENABLE_COMPONENT_${COMPONENT_NAME} to ON")
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE "wmtk::${COMPONENT_NAME}")
    endforeach()
endfunction()

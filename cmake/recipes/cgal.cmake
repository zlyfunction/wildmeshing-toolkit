# CGAL (https://github.com/CGAL/cgal)
# License: GPL/LGPL depending on components

if(TARGET CGAL::CGAL)
    return()
endif()

message(STATUS "Third-party: creating target 'CGAL::CGAL'")

include(CPM)

set(CGAL_HEADER_ONLY ON CACHE BOOL "Use CGAL header-only mode" FORCE)
set(CGAL_ENABLE_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
set(CGAL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CGAL_INSTALL_DOC OFF CACHE BOOL "" FORCE)
set(CGAL_INSTALL_MANUAL OFF CACHE BOOL "" FORCE)

set(CGAL_GIT_TAG v6.1 CACHE STRING "CGAL version to fetch")

CPMAddPackage(
    NAME CGAL
    GITHUB_REPOSITORY CGAL/cgal
    GIT_TAG ${CGAL_GIT_TAG}
    OPTIONS
        "CGAL_HEADER_ONLY ${CGAL_HEADER_ONLY}"
        "CGAL_ENABLE_PRECOMPILED_HEADERS ${CGAL_ENABLE_PRECOMPILED_HEADERS}"
        "WITH_CGAL_Qt5 OFF"
        "WITH_CGAL_ImageIO OFF"
        "WITH_GMP ON"
        "WITH_MPFR ON"
        "BUILD_TESTING ${CGAL_BUILD_TESTING}"
        "CGAL_INSTALL_DOC ${CGAL_INSTALL_DOC}"
        "CGAL_INSTALL_MANUAL ${CGAL_INSTALL_MANUAL}"
)

if(CGAL_ADDED)
    find_package(CGAL CONFIG REQUIRED HINTS ${CGAL_BINARY_DIR})
endif()

if(NOT TARGET CGAL::CGAL)
    if(NOT CGAL_INCLUDE_DIRS)
        message(FATAL_ERROR "Failed to locate CGAL include directories")
    endif()
    message(STATUS "Creating interface target CGAL::CGAL with ${CGAL_INCLUDE_DIRS}")
    add_library(CGAL::CGAL INTERFACE IMPORTED)
    target_include_directories(CGAL::CGAL INTERFACE ${CGAL_INCLUDE_DIRS})
    if(CGAL_3RD_PARTY_LIBRARIES)
        target_link_libraries(CGAL::CGAL INTERFACE ${CGAL_3RD_PARTY_LIBRARIES})
    endif()
    if(CGAL_3RD_PARTY_INCLUDE_DIRS)
        target_include_directories(CGAL::CGAL INTERFACE ${CGAL_3RD_PARTY_INCLUDE_DIRS})
    endif()
    if(CGAL_3RD_PARTY_DEFINITIONS)
        target_compile_definitions(CGAL::CGAL INTERFACE ${CGAL_3RD_PARTY_DEFINITIONS})
    endif()
endif()

set_target_properties(CGAL PROPERTIES FOLDER third_party)

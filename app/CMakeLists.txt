
include(wmtk_data)

option(WMTK_APP_TETWILD "tetwild application" OFF)
option(WMTK_APP_TRIWILD "triwild application" OFF)
option(WMTK_APP_HARMTET "Harmonic Triangulation app" OFF)
option(WMTK_APP_UNIFORM_REMESH "uniform remesh for surfaces" OFF)
option(WMTK_APP_SEC "surface shortest edge collapse" OFF)
option(WMTK_APP_QSLIM "surface qslim simplification" OFF)
option(WMTK_APP_UNIT_TESTS "unit tests for applications" OFF)
option(WMKT_APP_EXTREME_OPT "extreme opt" ON)
add_subdirectory(interior_tet_opt)

if(WMTK_APP_TETWILD)
    add_subdirectory(tetwild)
endif()

if(WMTK_APP_TRIWILD)
    add_subdirectory(triwild)
endif()

if(WMTK_APP_HARMTET)
    add_subdirectory(harmonic_tet)
endif()

if(WMTK_APP_UNIFORM_REMESH)
    add_subdirectory(remeshing)
endif()

if(WMTK_APP_SEC)
    add_subdirectory(shortest_edge_collapse)
endif()

if(WMTK_APP_QSLIM)
    add_subdirectory(qslim)
endif()

if(WMKT_APP_EXTREME_OPT)
    add_subdirectory(extreme_opt)
endif()

# ###############################################################################
# Tests
# ###############################################################################
if(WMTK_APP_UNIT_TESTS)
    add_subdirectory(tests)
endif()
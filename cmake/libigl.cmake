if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/zlyfunction/libigl.git
    GIT_TAG c0b9197c858471a49177486584418791ef1b5cde
)
FetchContent_MakeAvailable(libigl)
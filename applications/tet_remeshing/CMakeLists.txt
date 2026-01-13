include(wmtk_add_application)
wmtk_add_application(tet_remeshing_app
    tet_remeshing_main.cpp
    tet_remeshing_spec.hpp
    )
target_link_libraries(tet_remeshing_app PRIVATE
wmtk::input
wmtk::tet_remeshing
wmtk::output
nlohmann_json::nlohmann_json
)

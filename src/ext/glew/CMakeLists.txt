target_include_directories(toy_exe PRIVATE
    include)
target_compile_definitions(toy_exe PRIVATE
    -DGLEW_STATIC)
target_sources(toy_exe PRIVATE 
    glew.c
    )
add_subdirectory(include/GL)
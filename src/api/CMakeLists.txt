
add_library(migraphx_c
    api.cpp
)
set_target_properties(migraphx_c PROPERTIES EXPORT_NAME c)
rocm_set_soversion(migraphx_c 3.0)

rocm_clang_tidy_check(migraphx_c)
target_link_libraries(migraphx_c PRIVATE migraphx migraphx_tf migraphx_onnx migraphx_all_targets)

rocm_install_targets(
  TARGETS migraphx_c
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

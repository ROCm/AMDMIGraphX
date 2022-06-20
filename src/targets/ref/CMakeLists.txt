
add_library(migraphx_ref
    target.cpp
    lowering.cpp
    gemm.cpp
)
set_target_properties(migraphx_ref PROPERTIES EXPORT_NAME ref)
rocm_set_soversion(migraphx_ref ${MIGRAPHX_SO_VERSION})

find_path(BLAZE_INCLUDE blaze/Blaze.h)
find_package(Threads)

rocm_clang_tidy_check(migraphx_ref)
target_link_libraries(migraphx_ref migraphx Threads::Threads)
target_include_directories(migraphx_ref PRIVATE ${BLAZE_INCLUDE})
target_compile_definitions(migraphx_ref PRIVATE -DBLAZE_USE_CPP_THREADS)

rocm_install_targets(
  TARGETS migraphx_ref
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)



add_library(migraphx_cpu
    target.cpp
    lowering.cpp
    gemm.cpp
)
set_target_properties(migraphx_cpu PROPERTIES EXPORT_NAME cpu)
rocm_set_soversion(migraphx_cpu ${PROJECT_VERSION})

find_path(BLAZE_INCLUDE blaze/Blaze.h)
find_package(Threads)

rocm_clang_tidy_check(migraphx_cpu)
target_link_libraries(migraphx_cpu migraphx Threads::Threads)
target_include_directories(migraphx_cpu PRIVATE ${BLAZE_INCLUDE})
target_compile_definitions(migraphx_cpu PRIVATE -DBLAZE_USE_CPP_THREADS)

rocm_install_targets(
  TARGETS migraphx_cpu
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


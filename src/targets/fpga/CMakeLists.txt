
add_library(migraphx_fpga
    lowering.cpp
    target.cpp
)
set_target_properties(migraphx_fpga PROPERTIES EXPORT_NAME fpga)
rocm_set_soversion(migraphx_fpga ${MIGRAPHX_SO_VERSION})

find_path(BLAZE_INCLUDE blaze/Blaze.h)
find_package(Threads)

rocm_clang_tidy_check(migraphx_fpga)
target_link_libraries(migraphx_fpga migraphx Threads::Threads)
target_include_directories(migraphx_fpga PRIVATE ${BLAZE_INCLUDE})
target_compile_definitions(migraphx_fpga PRIVATE -DBLAZE_USE_CPP_THREADS)

target_link_libraries(migraphx_all_targets INTERFACE migraphx_fpga)

rocm_install_targets(
  TARGETS migraphx_fpga
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)



add_library(migraph_cpu
    target.cpp
    lowering.cpp
    gemm.cpp
)

find_path(BLAZE_INCLUDE blaze/Blaze.h)
find_package(Threads)

rocm_clang_tidy_check(migraph_cpu)
target_link_libraries(migraph_cpu migraph Threads::Threads)
target_include_directories(migraph_cpu PRIVATE ${BLAZE_INCLUDE})
target_compile_definitions(migraph_cpu PRIVATE -DBLAZE_USE_CPP_THREADS)

rocm_install_targets(
  TARGETS migraph_cpu
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


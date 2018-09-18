
add_library(migraph_cpu
    cpu_target.cpp
    cpu_lowering.cpp
    gemm.cpp
)

find_path(BLAZE_INCLUDE blaze/Blaze.h)
find_package(Threads)

rocm_clang_tidy_check(migraph_cpu)
target_link_libraries(migraph_cpu migraph Threads::Threads)
target_include_directories(migraph_cpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(migraph_cpu PRIVATE ${BLAZE_INCLUDE})
target_compile_definitions(migraph_cpu PRIVATE -DBLAZE_USE_CPP_THREADS)

install (TARGETS migraph_cpu
  LIBRARY DESTINATION /opt/rocm/lib)
install (DIRECTORY include/migraph DESTINATION /opt/rocm/include)


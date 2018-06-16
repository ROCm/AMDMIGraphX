
add_library(rtg_cpu SHARED
    cpu_target.cpp
)
rocm_clang_tidy_check(rtg_cpu)
target_link_libraries(rtg_cpu rtg)
target_include_directories(rtg_cpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install (TARGETS rtg_cpu
  LIBRARY DESTINATION /opt/rocm/lib)
install (DIRECTORY include/rtg DESTINATION /opt/rocm/include)

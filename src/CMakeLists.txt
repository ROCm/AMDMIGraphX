
add_library(rtg SHARED
    program.cpp
    shape.cpp
)
rocm_clang_tidy_check(rtg)
target_include_directories(rtg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_subdirectory(onnx)
add_subdirectory(targets/cpu)


install (TARGETS rtg
   LIBRARY DESTINATION /opt/rocm/lib)

install (DIRECTORY include/rtg DESTINATION /opt/rocm/include)


if(RTG_ENABLE_MIOPEN)
add_subdirectory(targets/miopen)
endif()


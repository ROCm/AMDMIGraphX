
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

add_library(migraph_device 
    hip_contiguous.cpp
)
rocm_clang_tidy_check(migraph_device)
target_link_libraries(migraph_device migraph hip::device)
target_include_directories(migraph_device PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_library(migraph_miopen
    hip.cpp
    miopen_target.cpp
)
rocm_clang_tidy_check(migraph_miopen)
target_link_libraries(migraph_miopen migraph MIOpen migraph_device)
target_include_directories(migraph_miopen PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

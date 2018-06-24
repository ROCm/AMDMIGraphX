
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

add_library(rtg_miopen
    hip.cpp
    miopen_target.cpp
)
rocm_clang_tidy_check(rtg_miopen)
target_link_libraries(rtg_miopen rtg MIOpen)
target_include_directories(rtg_miopen PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

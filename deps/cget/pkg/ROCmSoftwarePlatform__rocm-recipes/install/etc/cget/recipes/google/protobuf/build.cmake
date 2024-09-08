cmake_minimum_required( VERSION 3.11 )

if(BUILD_TESTING)
    set(protobuf_BUILD_TESTS ON CACHE BOOL "")
else()
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "")
endif()
add_subdirectory(cmake)

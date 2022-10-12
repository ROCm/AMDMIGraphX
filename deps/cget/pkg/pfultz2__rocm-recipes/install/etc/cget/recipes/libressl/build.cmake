cmake_minimum_required(VERSION 2.8)

find_package(cget-recipe-utils)

if(WIN32)
    file(WRITE include/syslog.h "")
    file(GLOB_RECURSE CMAKE_LISTS_FILES CMakeLists.txt)
    # TODO: This probably needs to be patched when cross-compiling
    foreach(FILE ${CMAKE_LISTS_FILES})
        patch_file(${FILE} "CMAKE_HOST_" "")
        patch_file(${FILE} "Ws2_32" "ws2_32")
    endforeach()
    patch_file(${CGET_CMAKE_ORIGINAL_SOURCE_FILE} "CMAKE_HOST_" "")
endif()

include(${CGET_CMAKE_ORIGINAL_SOURCE_FILE})
message("OPENSSL_LIBS: ${OPENSSL_LIBS}")

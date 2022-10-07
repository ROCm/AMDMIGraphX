

project(require-hcc)

if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    message(STATUS "HCC Compiler found")
else()
    message(FATAL_ERROR "HCC Compiler not set")
endif()

install(CODE "message(\"Installing nothing\")")

cmake_minimum_required(VERSION 3.5)
project (Test_migraphx_c_api)

set (MIGRAPHX_FOLDER $HOME/AMDMIGraphX)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)

set (CMAKE_CXX_STANDARD 14)

include_directories(../../src/api/include)
link_directories(../../lib)

file(GLOB util_SRC "*.cpp")
file(GLOB test_SRC test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_SRC})
add_library(utils SHARED ${util_SRC})

set(test_examples test_load_onnx.cpp)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} migraphx_c utils)
endforeach(filepath_name)


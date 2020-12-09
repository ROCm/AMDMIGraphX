cmake_minimum_required(VERSION 3.5)
project (PLS)

set (MIGRAPHX_FOLDER AMDMIGraphX)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)

set (CMAKE_CXX_STANDARD 14)

include_directories(../../../${MIGRAPHX_FOLDER}/src/api/include)
link_directories(../../../${MIGRAPHX_BUILD}/lib)

set (parse_load_save parse_load_save.cpp)

message("source file: " parse_load_save.cpp " ---> bin: " parse_load_save)
add_executable(parse_load_save parse_load_save.cpp)
target_link_libraries(parse_load_save migraphx_c)

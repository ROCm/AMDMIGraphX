
cmake_policy(SET CMP0057 NEW)

include(CTest)

find_package(Threads REQUIRED)
include(ProcessorCount)
ProcessorCount(N)
set(CTEST_PARALLEL_LEVEL ${N} CACHE STRING "CTest parallel level")
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j ${CTEST_PARALLEL_LEVEL} -C ${CMAKE_CFG_INTDIR} --timeout 1500)
add_custom_target(tests)

find_program(MIGRAPHX_GDB gdb)

if(MIGRAPHX_GDB)
    set(MIGRAPHX_TEST_GDB On CACHE BOOL "")
else()
    set(MIGRAPHX_TEST_GDB Off CACHE BOOL "")
endif()

set(SKIP_TESTS)

function(add_test_command NAME EXE)
    if(NAME IN_LIST SKIP_TESTS)
        add_test(NAME ${NAME} COMMAND echo skipped)
        set_tests_properties(${NAME} PROPERTIES DISABLED On)
    elseif(WIN32)
        set(WINPATH)
        foreach(PATH ${CMAKE_FIND_ROOT_PATH})
            list(APPEND WINPATH ${PATH}/bin)
        endforeach()
        file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmd"
            CONTENT "set PATH=${WINPATH};%PATH%
                    %1 ${ARGN}")
        add_test(NAME ${NAME} COMMAND ${WINE_CMD} cmd /c "${CMAKE_CURRENT_BINARY_DIR}/test_${NAME}.cmd" $<TARGET_FILE:${EXE}>)
    else()
        if(MIGRAPHX_TEST_GDB)
            # add_test(NAME ${NAME} COMMAND ${MIGRAPHX_GDB} 
            #     --batch
            #     --return-child-result
            #     -ex "set disable-randomization off"
            #     -ex run
            #     -ex backtrace
            #     --args $<TARGET_FILE:${EXE}> ${ARGN})
            set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/gdb/test_${NAME})
            file(MAKE_DIRECTORY ${TEST_DIR})
            file(GENERATE OUTPUT "${TEST_DIR}/run.cmake"
                CONTENT "
                # Remove previous core dump
                file(REMOVE ${TEST_DIR}/core)
                execute_process(COMMAND $<TARGET_FILE:${EXE}> ${ARGN} WORKING_DIRECTORY ${TEST_DIR} RESULT_VARIABLE RESULT)
                if(NOT RESULT EQUAL 0)
                    # TODO: check for core files based on pid when setting /proc/sys/kernel/core_uses_pid
                    if(EXISTS ${TEST_DIR}/core)
                        set(\$ENV{UBSAN_OPTIONS} print_stacktrace=1)
                        set(\$ENV{ASAN_OPTIONS} print_stacktrace=1)
                        execute_process(COMMAND ${MIGRAPHX_GDB} $<TARGET_FILE:${EXE}> ${TEST_DIR}/core -batch -ex bt)
                    endif()
                    message(FATAL_ERROR \"Test failed\")
                endif()
            ")
            add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} -P "${TEST_DIR}/run.cmake")
        else()
            add_test(NAME ${NAME} COMMAND ${EXE} ${ARGN})
        endif()
    endif()
endfunction()

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})
    # Cmake does not add flags correctly for gcc
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()
    separate_arguments(MIOPEN_TEST_FLAGS_ARGS UNIX_COMMAND ${MIOPEN_TEST_FLAGS})
    if(MIOPEN_TEST_ALL)
        set(TEST_COMMAND ${TEST_NAME} ${MIOPEN_TEST_FLOAT_ARG} --all ${MIOPEN_TEST_FLAGS_ARGS})
    else()
        set(TEST_COMMAND ${TEST_NAME} ${MIOPEN_TEST_FLOAT_ARG} ${MIOPEN_TEST_FLAGS_ARGS})
    endif()
    add_test_command(${TEST_NAME} ${TEST_COMMAND})
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
    target_link_libraries(${TEST_NAME} migraphx migraphx_cpu migraphx_onnx)
    target_include_directories(${TEST_NAME} PUBLIC include)
endfunction(add_test_executable)

file(GLOB TESTS *.cpp)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
    rocm_clang_tidy_check(test_${BASE_NAME})
endforeach()

if(MIGRAPHX_ENABLE_GPU)
    # gpu tests
    file(GLOB GPU_TESTS gpu/*.cpp)

    foreach(TEST ${GPU_TESTS})
        get_filename_component(BASE_NAME ${TEST} NAME_WE)
        add_test_executable(test_gpu_${BASE_NAME} ${TEST})
        rocm_clang_tidy_check(test_gpu_${BASE_NAME})
        set_tests_properties(test_gpu_${BASE_NAME} PROPERTIES 
            COST 10 
            RESOURCE_LOCK gpu
        )
        target_link_libraries(test_gpu_${BASE_NAME} migraphx_gpu)
    endforeach()
endif()

# Onnx test
set(TEST_ONNX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnx)
file (GLOB ONNX_TESTS ${TEST_ONNX_DIR}/*.cpp)
foreach(ONNX_TEST ${ONNX_TESTS})
    get_filename_component(BASE_NAME ${ONNX_TEST} NAME_WE)
    set(TEST_NAME test_${BASE_NAME})
    add_executable(${TEST_NAME} ${TES_ONNX_DIR}/${ONNX_TEST})
    rocm_clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} migraphx_onnx migraphx_cpu)
    target_include_directories(${TEST_NAME} PUBLIC include)
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/onnx) 
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
endforeach()

# tf test
add_executable(test_tf tf/tf_test.cpp)
rocm_clang_tidy_check(test_tf)
target_link_libraries(test_tf migraphx_tf migraphx_cpu)
target_include_directories(test_tf PUBLIC include)
add_test(NAME test_tf COMMAND $<TARGET_FILE:test_tf> WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tf) 
add_dependencies(tests test_tf)
add_dependencies(check test_tf)

if(MIGRAPHX_ENABLE_PYTHON)
add_subdirectory(py)
endif()

function(test_header NAME HEADER)

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/header-main-include-${NAME}.cpp 
        "#include <${HEADER}>\nint main() {}\n"
    )
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/header-static-include-${NAME}.cpp 
        "#include <${HEADER}>\n"
    )
    add_test_executable(${NAME}
        ${CMAKE_CURRENT_BINARY_DIR}/header-main-include-${NAME}.cpp 
        ${CMAKE_CURRENT_BINARY_DIR}/header-static-include-${NAME}.cpp
    )
endfunction()

function(test_headers PREFIX)
    file(GLOB HEADERS ${ARGN})

    foreach(HEADER ${HEADERS})
        file(RELATIVE_PATH HEADER_REL ${CMAKE_SOURCE_DIR} ${HEADER})
        string(MAKE_C_IDENTIFIER ${HEADER_REL} TEST_NAME)
        get_filename_component(BASE_NAME ${HEADER} NAME_WE)
        test_header(header_${TEST_NAME} ${PREFIX}/${BASE_NAME}.hpp)
        if(MIGRAPHX_ENABLE_GPU)
            target_link_libraries(header_${TEST_NAME} migraphx_gpu)
        endif()
    endforeach()
endfunction()

test_headers(migraphx ${CMAKE_SOURCE_DIR}/src/include/migraphx/*.hpp)
test_headers(migraphx/cpu ${CMAKE_SOURCE_DIR}/src/targets/cpu/include/migraphx/cpu/*.hpp)
if(MIGRAPHX_ENABLE_GPU)
test_headers(migraphx/gpu ${CMAKE_SOURCE_DIR}/src/targets/gpu/include/migraphx/gpu/*.hpp)
endif()

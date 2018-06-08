
cmake_policy(SET CMP0057 NEW)

include(CTest)

find_package(Threads REQUIRED)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

find_program(RTG_GDB gdb)

if(RTG_GDB)
    set(RTG_TEST_GDB On CACHE BOOL "")
else()
    set(RTG_TEST_GDB Off CACHE BOOL "")
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
        if(RTG_TEST_GDB)
            add_test(NAME ${NAME} COMMAND ${RTG_GDB} --batch --return-child-result -ex "set disable-randomization off" -ex r -ex bt --args $<TARGET_FILE:${EXE}> ${ARGN})
        else()
            add_test(NAME ${NAME} COMMAND ${EXE} ${ARGN})
        endif()
    endif()
endfunction()

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    rocm_clang_tidy_check(${TEST_NAME})
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
    target_link_libraries(${TEST_NAME} rtg rtg_cpu)
    target_include_directories(${TEST_NAME} PUBLIC include)
endfunction(add_test_executable)

file(GLOB TESTS *.cpp)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_test_executable(test_${BASE_NAME} ${TEST})
endforeach()

if(RTG_ENABLE_MIOPEN)
    # miopen tests
    file(GLOB MIOPEN_TESTS miopen/*.cpp)

    foreach(TEST ${MIOPEN_TESTS})
        get_filename_component(BASE_NAME ${TEST} NAME_WE)
        add_test_executable(test_miopen_${BASE_NAME} ${TEST})
        target_link_libraries(test_miopen_${BASE_NAME} rtg_miopen)
    endforeach()
endif()

include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(CTest)

find_package(Threads REQUIRED)
include(ProcessorCount)
ProcessorCount(_rocm_ctest_parallel_level)
set(CTEST_PARALLEL_LEVEL ${_rocm_ctest_parallel_level} CACHE STRING "CTest parallel level")
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j ${CTEST_PARALLEL_LEVEL} -C ${CMAKE_CFG_INTDIR} --timeout 5000)
add_custom_target(tests COMMENT "Build all tests.")
add_dependencies(check tests)

# rocm_package_setup_component(test DEPENDS COMPONENT runtime)

define_property(TARGET PROPERTY "ROCM_TEST_INSTALLDIR" BRIEF_DOCS "Install dir for tests" FULL_DOCS "Install dir for tests")

# TODO: Move to ROCMInstallTargets
define_property(TARGET PROPERTY "ROCM_INSTALL_DIR" BRIEF_DOCS "Install dir for target" FULL_DOCS "Install dir for target")
function(rocm_set_install_dir_property)
    set(options)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs TARGETS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(PARSE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown keywords given to rocm_set_install_dir_property(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()

    if(PARSE_DESTINATION MATCHES "^/|$")
        set_target_properties(${PARSE_TARGETS} PROPERTIES ROCM_INSTALL_DIR ${PARSE_DESTINATION})
    else()
        set_target_properties(${PARSE_TARGETS} PROPERTIES ROCM_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${PARSE_DESTINATION})
    endif()
endfunction()

add_library(rocm_test_dependencies INTERFACE)
function(rocm_test_link_libraries)
    target_link_libraries(rocm_test_dependencies INTERFACE ${ARGN})
endfunction()
function(rocm_test_include_directories)
    target_include_directories(rocm_test_dependencies INTERFACE ${ARGN})
endfunction()

find_program(ROCM_GDB gdb)

if(ROCM_GDB)
    set(ROCM_TEST_GDB On CACHE BOOL "")
else()
    set(ROCM_TEST_GDB Off CACHE BOOL "")
endif()

set(_rocm_test_config_content "")

set(_rocm_test_package_dir ${CMAKE_BINARY_DIR}/rocm-test-package)
set(_rocm_test_config_file ${_rocm_test_package_dir}/CTestTestfile.cmake)
set(_rocm_test_run_save_tests ${_rocm_test_package_dir}/run-save-tests.cmake)
file(MAKE_DIRECTORY ${_rocm_test_package_dir})
file(WRITE ${_rocm_test_run_save_tests} "")

function(rocm_save_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs COMMAND)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(PARSE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown keywords given to rocm_save_test(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()
    if(NOT PARSE_NAME)
        message(FATAL_ERROR "Missing NAME in rocm_save_test()")
    endif()
    if(NOT PARSE_COMMAND)
        message(FATAL_ERROR "Missing COMMAND in rocm_save_test()")
    endif()
    set(COMMAND "")
    foreach(ARG ${PARSE_COMMAND})
        if(TARGET ${ARG})
            string(APPEND COMMAND " \"$<GENEX_EVAL:$<TARGET_PROPERTY:${ARG},ROCM_INSTALL_DIR>>/$<TARGET_FILE_NAME:${ARG}>\"")
        else()
            string(APPEND COMMAND " \"${ARG}\"")
        endif()
    endforeach()
    file(APPEND ${_rocm_test_config_file}.in "add_test(${PARSE_NAME} ${COMMAND})\n")
    set(PROP_NAMES
        ATTACHED_FILES
        ATTACHED_FILES_ON_FAIL
        COST
        DEPENDS
        DISABLED
        ENVIRONMENT
        ENVIRONMENT_MODIFICATION
        FAIL_REGULAR_EXPRESSION
        FIXTURES_CLEANUP
        FIXTURES_REQUIRED
        FIXTURES_SETUP
        LABELS
        MEASUREMENT
        PASS_REGULAR_EXPRESSION
        PROCESSOR_AFFINITY
        PROCESSORS
        REQUIRED_FILES
        RESOURCE_GROUPS
        RESOURCE_LOCK
        RUN_SERIAL
        SKIP_REGULAR_EXPRESSION
        SKIP_RETURN_CODE
        TIMEOUT
        TIMEOUT_AFTER_MATCH
        WILL_FAIL
        WORKING_DIRECTORY
    )
    set(PROPS "")
    foreach(PROPERTY ${PROP_NAMES})
        get_test_property(${PARSE_NAME} ${PROPERTY} VALUE)
        if(VALUE)
            string(APPEND PROPS " ${PROPERTY} \"${VALUE}\"")
        endif()
    endforeach()
    if(PROPS)
        file(APPEND ${_rocm_test_config_file}.in "set_tests_properties(${PARSE_NAME} PROPERTIES ${PROPS})\n")
    endif()
endfunction()

function(rocm_add_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs COMMAND)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(PARSE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown keywords given to rocm_add_test(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()
    if(NOT PARSE_NAME)
        message(FATAL_ERROR "Missing NAME in rocm_add_test()")
    endif()
    if(NOT PARSE_COMMAND)
        message(FATAL_ERROR "Missing COMMAND in rocm_add_test()")
    endif()
    file(APPEND ${_rocm_test_run_save_tests} "rocm_save_test(NAME ${PARSE_NAME} COMMAND ${PARSE_COMMAND})\n")

    set(COMMAND ${PARSE_COMMAND})
    list(GET COMMAND 0 COMMAND_EXE)
    set(COMMAND_ARGS ${COMMAND})
    list(POP_FRONT COMMAND_ARGS)


    if(ROCM_TEST_GDB AND TARGET ${COMMAND_EXE})
        # add_test(NAME ${NAME} COMMAND ${ROCM_GDB} 
        #     --batch
        #     --return-child-result
        #     -ex "set disable-randomization off"
        #     -ex run
        #     -ex backtrace
        #     --args $<TARGET_FILE:${EXE}> ${ARGN})
        set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/gdb/test_${PARSE_NAME})
        file(MAKE_DIRECTORY ${TEST_DIR})
        if (NOT EXISTS ${TEST_DIR})
            message(FATAL_ERROR "Failed to create test directory: ${TEST_DIR}")
        endif()
        file(GENERATE OUTPUT "${TEST_DIR}/run.cmake"
            CONTENT "
            # Remove previous core dump
            file(REMOVE ${TEST_DIR}/core)
            execute_process(COMMAND $<TARGET_FILE:${COMMAND_EXE}> ${COMMAND_ARGS} WORKING_DIRECTORY ${TEST_DIR} RESULT_VARIABLE RESULT)
            if(NOT RESULT EQUAL 0)
                # TODO: check for core files based on pid when setting /proc/sys/kernel/core_uses_pid
                if(EXISTS ${TEST_DIR}/core)
                    set(\$ENV{UBSAN_OPTIONS} print_stacktrace=1)
                    set(\$ENV{ASAN_OPTIONS} print_stacktrace=1)
                    execute_process(COMMAND ${ROCM_GDB} $<TARGET_FILE:${COMMAND_EXE}> ${TEST_DIR}/core -batch -ex bt)
                endif()
                message(FATAL_ERROR \"Test failed\")
            endif()
        ")
        set(COMMAND ${CMAKE_COMMAND} -P "${TEST_DIR}/run.cmake")
    endif()
    add_test(NAME ${PARSE_NAME} COMMAND ${COMMAND})
    set_tests_properties(${PARSE_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endfunction()

function(rocm_mark_as_test)
    foreach(TEST_TARGET ${ARGN})
        get_target_property(TEST_TARGET_TYPE ${TEST_TARGET} TYPE)
        # We can only use EXCLUDE_FROM_ALL on build targets
        if(NOT "${TEST_TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
            set_target_properties(${TEST_TARGET}
                PROPERTIES EXCLUDE_FROM_ALL TRUE
            )
            target_link_libraries(${TEST_TARGET} rocm_test_dependencies)
        endif()
        add_dependencies(tests ${TEST_TARGET})
    endforeach()
endfunction()

function(rocm_link_test_dependencies)
    foreach(TEST_TARGET ${ARGN})
        get_target_property(TEST_TARGET_TYPE ${TEST_TARGET} TYPE)
        # We can only use target_link_libraries on build targets
        if(NOT "${TEST_TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
            target_link_libraries(${TEST_TARGET} rocm_test_dependencies)
        endif()
    endforeach()
endfunction()

function(rocm_install_test)
    set(options)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs TARGETS FILES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(PARSE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown keywords given to rocm_install_test(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()
    set(INSTALL_PREFIX "$<TARGET_PROPERTY:tests,ROCM_TEST_INSTALLDIR>")
    if(PARSE_TARGETS)
        install(TARGETS ${PARSE_TARGETS} COMPONENT test DESTINATION ${INSTALL_PREFIX}/bin)
        rocm_set_install_dir_property(TARGETS ${PARSE_TARGETS} DESTINATION ${INSTALL_PREFIX}/bin)
        get_target_property(INSTALLDIR ${PARSE_TARGETS} ROCM_INSTALL_DIR)
    endif()
    if(PARSE_FILES)
        install(FILES ${PARSE_FILES} COMPONENT test DESTINATION ${INSTALL_PREFIX}/${PARSE_DESTINATION})
    endif()
endfunction()

function(rocm_add_test_executable EXE)
    add_executable (${EXE} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${EXE} ${CMAKE_THREAD_LIBS_INIT})
    # Cmake does not add flags correctly for gcc
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
        set_target_properties(${EXE} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
    endif()
    rocm_mark_as_test(${EXE})
    rocm_link_test_dependencies(${EXE})
    rocm_add_test(NAME ${EXE} COMMAND ${EXE})
    rocm_install_test(TARGETS ${EXE})
endfunction()

function(rocm_test_header NAME HEADER)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/header-main-include-${NAME}.cpp 
        "#include <${HEADER}>\nint main() {}\n"
    )
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/header-static-include-${NAME}.cpp 
        "#include <${HEADER}>\n"
    )
    rocm_add_test_executable(${NAME}
        ${CMAKE_CURRENT_BINARY_DIR}/header-main-include-${NAME}.cpp 
        ${CMAKE_CURRENT_BINARY_DIR}/header-static-include-${NAME}.cpp
    )
endfunction()

function(rocm_test_install_ctest)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(PARSE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown keywords given to rocm_test_install_ctest(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()

    set_target_properties(tests PROPERTIES ROCM_TEST_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/share/test/${PARSE_NAME})
    file(WRITE ${_rocm_test_config_file}.in "")
    include(${_rocm_test_run_save_tests})
    file(GENERATE OUTPUT ${_rocm_test_config_file} INPUT ${_rocm_test_config_file}.in)
    rocm_install_test(FILES ${_rocm_test_config_file})
endfunction()

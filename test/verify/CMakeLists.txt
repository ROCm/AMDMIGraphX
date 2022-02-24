
file(GLOB VERIFY_TESTS ${CONFIGURE_DEPENDS} *.cpp)

add_executable(test_verify ${VERIFY_TESTS})
add_dependencies(tests test_verify)
add_dependencies(check test_verify)
target_link_libraries(test_verify migraphx migraphx_all_targets)
target_include_directories(test_verify PUBLIC ../include)
rocm_clang_tidy_check(test_verify)

foreach(SECTION general rnn)
    add_test_command(test_verify_${SECTION} test_verify ${SECTION})
    set_tests_properties(test_verify_${SECTION} PROPERTIES 
        COST 100 
    )
    if(MIGRAPHX_ENABLE_GPU)
        set_tests_properties(test_verify_${SECTION} PROPERTIES 
            RESOURCE_LOCK gpu
        )
    endif()
endforeach()

rocm_test_link_libraries(pocket_rocm_cxx)
rocm_test_include_directories(include)

function(add_tests PREFIX GLOB)
    file(GLOB TESTS CONFIGURE_DEPENDS ${GLOB})
    foreach(TEST ${TESTS})
        get_filename_component(BASE_NAME ${TEST} NAME_WE)
        rocm_add_test_executable(test_${PREFIX}_${BASE_NAME} ${TEST})
        rocm_clang_tidy_check(test_${PREFIX}_${BASE_NAME})
    endforeach()
endfunction()

add_tests(type_traits type_traits/*.cpp)


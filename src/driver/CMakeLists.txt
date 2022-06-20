
add_executable(driver 
    main.cpp
    verify.cpp
    perf.cpp
    resnet50.cpp
    inceptionv3.cpp
    alexnet.cpp
    marker_roctx.cpp
)
set_target_properties(driver PROPERTIES OUTPUT_NAME migraphx-driver)
# Copy driver for backwards compatibility
add_custom_command(
        TARGET driver 
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:driver>
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/driver
        BYPRODUCTS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/driver
)
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/driver)
rocm_clang_tidy_check(driver)
target_link_libraries(driver migraphx_all_targets migraphx_onnx migraphx_tf)

rocm_install_targets(
  TARGETS driver
)

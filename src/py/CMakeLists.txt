
option(MIGRAPHX_ENABLE_PYTHON "Enable python bindings" ON)
if(MIGRAPHX_ENABLE_PYTHON)
    find_program(DEFAULT_PYTHON_EXE python)
    if(DEFAULT_PYTHON_EXE)
        set(PYTHON_EXECUTABLE ${DEFAULT_PYTHON_EXE} CACHE PATH "Path to python executable")
    endif()
    find_package(pybind11 REQUIRED)
    pybind11_add_module(migraphx_py migraphx_py.cpp)
    set_target_properties(migraphx_py PROPERTIES 
        OUTPUT_NAME migraphx
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    if(MIGRAPHX_ENABLE_TF)
        target_link_libraries(migraphx_py PRIVATE migraphx migraphx_tf migraphx_cpu)
        target_compile_definitions(migraphx_py PRIVATE -DENABLE_TF)
    else()
        target_link_libraries(migraphx_py PRIVATE migraphx migraphx_onnx migraphx_cpu)
    endif()
    if(MIGRAPHX_ENABLE_GPU)
        target_link_libraries(migraphx_py PRIVATE migraphx_gpu)
        target_compile_definitions(migraphx_py PRIVATE -DHAVE_GPU)
    endif()
    rocm_install_targets(TARGETS migraphx_py)
endif()

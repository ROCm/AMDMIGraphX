
option(MIGRAPHX_ENABLE_PYTHON "Enable python bindings" ON)
if(MIGRAPHX_ENABLE_PYTHON)
    include(PythonModules)

    add_custom_target(migraphx_py)

    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        py_add_module(migraphx_py_${PYTHON_VERSION} migraphx_py.cpp PYTHON_VERSION ${PYTHON_VERSION} PYTHON_MODULE migraphx)
        target_link_libraries(migraphx_py_${PYTHON_VERSION} PRIVATE migraphx migraphx_tf migraphx_onnx migraphx_all_targets)
        rocm_install_targets(TARGETS migraphx_py_${PYTHON_VERSION})
        add_dependencies(migraphx_py migraphx_py_${PYTHON_VERSION})
    endforeach()
endif()

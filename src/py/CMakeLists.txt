
find_program(DEFAULT_PYTHON_EXE python)
if(DEFAULT_PYTHON_EXE)
set(PYTHON_EXECUTABLE ${DEFAULT_PYTHON_EXE} CACHE PATH "Path to python executable")
endif()
find_package(pybind11 REQUIRED)
pybind11_add_module(migraphx_py migraphx_py.cpp)
target_link_libraries(migraphx_py migraphx migraphx_onnx migraphx_cpu)
set_target_properties(migraphx_py PROPERTIES 
    OUTPUT_NAME migraphx
)
if(MIGRAPHX_ENABLE_GPU)
target_link_libraries(migraphx_py migraphx_gpu)
target_compile_definitions(migraphx_py PRIVATE -DHAVE_GPU)
endif()

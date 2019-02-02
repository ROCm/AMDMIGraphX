
find_package(pybind11 REQUIRED)
pybind11_add_module(migraphx_py migraphx_py.cpp)
target_link_libraries(migraphx_py migraphx migraphx_onnx migraphx_gpu migraphx_cpu)

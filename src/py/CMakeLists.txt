#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

add_library(migraphx_py py_loader.cpp)
migraphx_generate_export_header(migraphx_py)
target_include_directories(migraphx_py PRIVATE include ../common_api/ ../common_api/include)
target_link_libraries(migraphx_py PUBLIC migraphx_onnx migraphx_ref migraphx_c migraphx)
rocm_install_targets(TARGETS migraphx_py INCLUDE include)

include(PythonModules)
 
file(GLOB BINDING_SRCS migraphx_py.cpp common_api/*.cpp)

foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
    py_add_module(migraphx_pybind_${PYTHON_VERSION} ${BINDING_SRCS} PYTHON_VERSION ${PYTHON_VERSION} PYTHON_MODULE migraphx)
    target_include_directories(migraphx_pybind_${PYTHON_VERSION} PRIVATE include)
    target_link_libraries(migraphx_pybind_${PYTHON_VERSION} PRIVATE migraphx migraphx_tf migraphx_onnx migraphx_all_targets migraphx_ref migraphx_c)
    rocm_install_targets(TARGETS migraphx_pybind_${PYTHON_VERSION})
    add_dependencies(migraphx_py migraphx_pybind_${PYTHON_VERSION})

    add_library(migraphx_py_${PYTHON_VERSION} py.cpp)
    target_include_directories(migraphx_py_${PYTHON_VERSION} PRIVATE include)
    target_link_libraries(migraphx_py_${PYTHON_VERSION} PUBLIC migraphx)
    target_link_libraries(migraphx_py_${PYTHON_VERSION} PRIVATE pybind11::pybind11 python${PYTHON_VERSION}::runtime)
    rocm_install_targets(TARGETS migraphx_py_${PYTHON_VERSION})
    add_dependencies(migraphx_py migraphx_py_${PYTHON_VERSION})
endforeach()

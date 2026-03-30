#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

file(GLOB DXGML_SRCS CONFIGURE_DEPENDS *.cpp)
add_library(migraphx_dxgml ${DXGML_SRCS})
set_target_properties(migraphx_dxgml PROPERTIES EXPORT_NAME dxgml)
migraphx_generate_export_header(migraphx_dxgml)
rocm_set_soversion(migraphx_dxgml ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_dxgml)

# DxGML parser is a pure text-based parser — no MLIR C API or external dialect
# libraries are required.  The only dependency is the MIGraphX core library.
target_link_libraries(migraphx_dxgml PUBLIC migraphx)

rocm_install_targets(
  PRIVATE
  TARGETS migraphx_dxgml
)

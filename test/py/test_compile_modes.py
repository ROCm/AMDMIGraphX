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
import migraphx


def test_compile_modes_enum_exists():
    assert hasattr(migraphx, 'compile_modes')
    assert hasattr(migraphx.compile_modes, 'eager')
    assert hasattr(migraphx.compile_modes, 'balanced')
    assert hasattr(migraphx.compile_modes, 'max')


def test_compile_modes_enum_values():
    assert migraphx.compile_modes.eager.value == 0
    assert migraphx.compile_modes.balanced.value == 50
    assert migraphx.compile_modes.max.value == 100


def test_compile_with_eager_mode():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    s1 = p.get_output_shapes()[-1]
    p.compile(migraphx.get_target("ref"),
              compile_mode=migraphx.compile_modes.eager)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2


def test_compile_with_balanced_mode():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    s1 = p.get_output_shapes()[-1]
    p.compile(migraphx.get_target("ref"),
              compile_mode=migraphx.compile_modes.balanced)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2


def test_compile_with_max_mode():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    s1 = p.get_output_shapes()[-1]
    p.compile(migraphx.get_target("ref"),
              compile_mode=migraphx.compile_modes.max)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2


def test_compile_default_mode():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    s1 = p.get_output_shapes()[-1]
    # Default should be balanced
    p.compile(migraphx.get_target("ref"))
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2


def test_compile_eager_produces_valid_output():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    p.compile(migraphx.get_target("ref"),
              compile_mode=migraphx.compile_modes.eager)
    params = {}
    for key, value in p.get_parameter_shapes().items():
        params[key] = migraphx.generate_argument(value)
    result = p.run(params)
    assert len(result) > 0


test_compile_modes_enum_exists()
test_compile_modes_enum_values()
test_compile_with_eager_mode()
test_compile_with_balanced_mode()
test_compile_with_max_mode()
test_compile_default_mode()
test_compile_eager_produces_valid_output()

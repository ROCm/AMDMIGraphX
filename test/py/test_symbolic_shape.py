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


def test_create_symbolic_dyn_dims():
    sym = migraphx.shape.dynamic_dimension(migraphx.sym.var("n", 1, 4))
    assert sym.is_symbolic()
    assert not sym.is_fixed()
    assert sym.min == 1
    assert sym.max == 4

    sym_opt = migraphx.shape.dynamic_dimension(
        migraphx.sym.var("n", 1, 4, {2, 4}))
    assert sym_opt.is_symbolic()

    rng = migraphx.shape.dynamic_dimension(1, 4)
    assert not rng.is_symbolic()


def test_create_symbolic_compound_expr():
    n = migraphx.sym.var("n", 1, 8)
    product = migraphx.shape.dynamic_dimension(n * migraphx.sym.lit(3))
    assert product.is_symbolic()

    parsed = migraphx.shape.dynamic_dimension(migraphx.sym.parse("n + 1"))
    assert parsed.is_symbolic()


def test_create_symbolic_dyn_shape():
    dds = [
        migraphx.shape.dynamic_dimension(migraphx.sym.var("n", 1, 4)),
        migraphx.shape.dynamic_dimension(3, 3)
    ]
    s = migraphx.shape(type='float', dyn_dims=dds)
    assert s.dynamic()
    assert s.dyn_dims()[0].is_symbolic()
    assert not s.dyn_dims()[1].is_symbolic()


def test_parse_onnx_symbolic_dim_param():
    dim_params = {
        "dim0": migraphx.shape.dynamic_dimension(1, 8),
        "dim1": migraphx.shape.dynamic_dimension(2, 16)
    }
    p = migraphx.parse_onnx("dim_param_test.onnx",
                            use_symbolic_shapes=True,
                            dim_params=dim_params)
    s = p.get_parameter_shapes()["0"]
    assert s.dynamic()
    dd = s.dyn_dims()
    assert dd[0].is_symbolic()
    assert dd[1].is_symbolic()


def test_parse_onnx_symbolic_dyn_input():
    p = migraphx.parse_onnx(
        "dim_param_test.onnx",
        map_dyn_input_dims={
            "0": [
                migraphx.shape.dynamic_dimension(migraphx.sym.var("n", 1, 8)),
                migraphx.shape.dynamic_dimension(migraphx.sym.var("m", 2, 16))
            ]
        })
    s = p.get_parameter_shapes()["0"]
    assert s.dynamic()
    dd = s.dyn_dims()
    assert dd[0].is_symbolic()
    assert dd[1].is_symbolic()


def test_parse_onnx_symbolic_default_dyn_dim():
    p = migraphx.parse_onnx(
        "dim_param_test.onnx",
        use_symbolic_shapes=True,
        default_dyn_dim_value=migraphx.shape.dynamic_dimension(1, 8))
    s = p.get_parameter_shapes()["0"]
    assert s.dynamic()
    dd = s.dyn_dims()
    assert dd[0].is_symbolic()
    assert dd[1].is_symbolic()


if __name__ == "__main__":
    test_create_symbolic_dyn_dims()
    test_create_symbolic_compound_expr()
    test_create_symbolic_dyn_shape()
    test_parse_onnx_symbolic_dim_param()
    test_parse_onnx_symbolic_dyn_input()
    test_parse_onnx_symbolic_default_dyn_dim()

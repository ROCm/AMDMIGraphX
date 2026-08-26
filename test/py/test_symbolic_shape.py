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
    sym = migraphx.shape.dynamic_dimension(
        "n", {"n": migraphx.shape.dynamic_dimension(1, 4)})
    assert sym.is_symbolic()
    assert not sym.is_fixed()
    assert sym.min == 1
    assert sym.max == 4

    sym_opt = migraphx.shape.dynamic_dimension(
        "n", {"n": migraphx.shape.dynamic_dimension(1, 4, {2, 4})})
    assert sym_opt.is_symbolic()

    rng = migraphx.shape.dynamic_dimension(1, 4)
    assert not rng.is_symbolic()


def test_create_symbolic_compound_expr():
    product = migraphx.shape.dynamic_dimension(
        "n * 3", {"n": migraphx.shape.dynamic_dimension(1, 8)})
    assert product.is_symbolic()

    parsed = migraphx.shape.dynamic_dimension(
        "n + 1", {"n": migraphx.shape.dynamic_dimension(1, 8)})
    assert parsed.is_symbolic()


def test_create_symbolic_dyn_shape():
    dds = [
        migraphx.shape.dynamic_dimension(
            "n", {"n": migraphx.shape.dynamic_dimension(1, 4)}),
        migraphx.shape.dynamic_dimension(3, 3)
    ]
    s = migraphx.shape(type='float', dyn_dims=dds)
    assert s.dynamic()
    assert s.dyn_dims()[0].is_symbolic()
    assert not s.dyn_dims()[1].is_symbolic()


def _symbolic_program():
    p = migraphx.program()
    m = p.get_main_module()
    s = migraphx.shape(type='float',
                       dyn_dims=[
                           migraphx.shape.dynamic_dimension(
                               "n * 3 + 1",
                               {"n": migraphx.shape.dynamic_dimension(1, 8)}),
                           migraphx.shape.dynamic_dimension(3, 3)
                       ])
    m.add_return(
        [m.add_instruction(migraphx.op("neg"), [m.add_parameter("x", s)])])
    return p


def test_to_py_preserves_symbolic_expression():
    p = _symbolic_program()
    assert "migraphx.shape.from_json" in p.to_py()

    # The generated code has to rebuild an equal program, expression included. Sorting first
    # because to_py does not preserve the declaration order of parameters.
    scope = {"migraphx": migraphx}
    exec(p.to_py(), scope)
    assert scope["p"].sort() == p.sort()


def test_parse_onnx_symbolic_dyn_input():
    p = migraphx.parse_onnx(
        "dim_param_test.onnx",
        map_dyn_input_dims={
            "0": [
                migraphx.shape.dynamic_dimension(
                    "n", {"n": migraphx.shape.dynamic_dimension(1, 8)}),
                migraphx.shape.dynamic_dimension(
                    "m", {"m": migraphx.shape.dynamic_dimension(2, 16)})
            ]
        })
    s = p.get_parameter_shapes()["0"]
    assert s.dynamic()
    dd = s.dyn_dims()
    assert dd[0].is_symbolic()
    assert dd[1].is_symbolic()


if __name__ == "__main__":
    test_create_symbolic_dyn_dims()
    test_create_symbolic_compound_expr()
    test_create_symbolic_dyn_shape()
    test_to_py_preserves_symbolic_expression()
    test_parse_onnx_symbolic_dyn_input()

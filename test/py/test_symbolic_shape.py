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


def test_create_symbolic_shape_from_strings():
    dd = migraphx.shape.dynamic_dimension
    s = migraphx.shape(type='float',
                       dyn_dims=["n", "3"],
                       symbols={"n": dd(1, 8, {2, 4})})
    assert s.symbolic()
    assert s.dyn_dims()[0].expression == "n"
    assert s.dyn_dims()[0].min == 1
    assert s.dyn_dims()[0].max == 8
    assert s.dyn_dims()[0].optimals == {2, 4}
    # An all-symbolic shape gets packed standard strides when none are given.
    assert s.standard()
    assert s.dyn_strides() == ["3", "1"]

    # A range-based dimension carries no expression.
    r = migraphx.shape(type='float', dyn_dims=[dd(1, 4)])
    assert r.dyn_dims()[0].expression is None


def test_create_symbolic_shape_compound_expression():
    dd = migraphx.shape.dynamic_dimension
    s = migraphx.shape(type='float',
                       dyn_dims=["3*n + 1"],
                       symbols={"n": dd(1, 8)})
    assert s.dyn_dims()[0].expression == "3*n + 1"
    assert (s.dyn_dims()[0].min, s.dyn_dims()[0].max) == (4, 25)


def test_create_symbolic_shape_with_strides():
    dd = migraphx.shape.dynamic_dimension
    s = migraphx.shape(type='float',
                       dyn_dims=["n", "3"],
                       symbols={"n": dd(1, 8)},
                       dyn_strides=["1", "n"])
    assert not s.standard()
    assert s.dyn_strides() == ["1", "n"]


def test_create_symbolic_shape_multiple_constraints():
    dd = migraphx.shape.dynamic_dimension
    # A list of bounds asserts several intervals, which is what merging two differently bounded
    # same-named variables produces.
    s = migraphx.shape(type='float',
                       dyn_dims=["n"],
                       symbols={"n": [dd(1, 20), dd(2, 10, {4})]})
    assert s.dyn_dims()[0].is_symbolic()
    assert s.dyn_dims()[0].optimals == {4}


def test_symbol_table_round_trip():
    dd = migraphx.shape.dynamic_dimension
    s = migraphx.shape(type='float',
                       dyn_dims=["3*n + 1", "m"],
                       symbols={
                           "n": dd(1, 8, {2, 4}),
                           "m": dd(2, 16)
                       })
    table = s.symbol_table()
    # A symbol always maps to a list here, even where the constructor also takes a bare
    # dynamic_dimension.
    assert sorted(table) == ["m", "n"]
    assert len(table["n"]) == 1
    assert (table["n"][0].min, table["n"][0].max) == (1, 8)
    assert table["n"][0].optimals == {2, 4}
    # The table is what the constructor takes, so a shape rebuilds from what can be read of it.
    assert migraphx.shape(type='float',
                          dyn_dims=["3*n + 1", "m"],
                          symbols=table) == s


def test_symbol_table_strides_and_multiple_constraints():
    dd = migraphx.shape.dynamic_dimension
    # A stride shares the dimensions' symbols, so the table does not grow for it.
    s = migraphx.shape(type='float',
                       dyn_dims=["n", "3"],
                       symbols={"n": dd(1, 8)},
                       dyn_strides=["1", "n"])
    assert list(s.symbol_table()) == ["n"]
    assert migraphx.shape(type='float',
                          dyn_dims=["n", "3"],
                          symbols=s.symbol_table(),
                          dyn_strides=s.dyn_strides()) == s

    m = migraphx.shape(type='float',
                       dyn_dims=["n"],
                       symbols={"n": [dd(1, 20), dd(2, 10, {4})]})
    assert len(m.symbol_table()["n"]) == 2
    assert migraphx.shape(type='float',
                          dyn_dims=["n"],
                          symbols=m.symbol_table()) == m


# A range-based or static shape names no symbols.
def test_symbol_table_empty_without_symbols():
    dd = migraphx.shape.dynamic_dimension
    assert migraphx.shape(type='float', dyn_dims=[dd(1,
                                                     4)]).symbol_table() == {}
    assert migraphx.shape(type='float', lens=[2, 3]).symbol_table() == {}


def test_symbol_name_must_be_an_identifier():
    dd = migraphx.shape.dynamic_dimension
    # A symbolic shape is spelled as an expression string, so a name has to survive parsing.
    try:
        migraphx.shape(type='float',
                       dyn_dims=["input.1"],
                       symbols={"input.1": dd(1, 8)})
    except RuntimeError:
        pass
    else:
        assert False, "expected a non-identifier symbol name to be rejected"


def test_to_py_preserves_symbolic_expression():
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

    code = p.to_py()
    # The second dimension is range-based, so this is the per-dimension spelling.
    assert not s.symbolic()
    assert '"3*n + 1"' in code

    # The generated code has to rebuild an equal program, expression included. sort() normalizes
    # instruction order, which to_py only perturbs once a module has more than one parameter.
    scope = {"migraphx": migraphx}
    exec(code, scope)
    assert scope["p"].sort() == p.sort()

    # Program equality compares printed IR, which renders neither optimals nor symbolic strides,
    # so the expression is only really pinned by comparing the shape itself.
    assert scope["p"].get_parameter_shapes()["x"] == s


def test_to_py_preserves_symbolic_strides():
    p = migraphx.program()
    m = p.get_main_module()
    s = migraphx.shape(type='float',
                       dyn_dims=["n", "3"],
                       symbols={"n": migraphx.shape.dynamic_dimension(1, 8)},
                       dyn_strides=["1", "n"])
    m.add_return(
        [m.add_instruction(migraphx.op("neg"), [m.add_parameter("x", s)])])

    code = p.to_py()
    assert "dyn_strides" in code
    scope = {"migraphx": migraphx}
    exec(code, scope)
    assert scope["p"].get_parameter_shapes()["x"] == s


# make_symbolic_shape cannot express a range dimension, so a partly symbolic shape is printed
# dimension by dimension instead.
def test_to_py_mixed_symbolic_and_range():
    p = migraphx.program()
    m = p.get_main_module()
    s = migraphx.shape(type='float',
                       dyn_dims=[
                           migraphx.shape.dynamic_dimension(
                               "n",
                               {"n": migraphx.shape.dynamic_dimension(1, 8)}),
                           migraphx.shape.dynamic_dimension(3, 5)
                       ])
    m.add_return(
        [m.add_instruction(migraphx.op("neg"), [m.add_parameter("x", s)])])

    assert not s.symbolic()
    scope = {"migraphx": migraphx}
    exec(p.to_py(), scope)
    assert scope["p"].get_parameter_shapes()["x"] == s


def test_to_py_preserves_dyn_dim_optimals():
    p = migraphx.program()
    m = p.get_main_module()
    s = migraphx.shape(type='float',
                       dyn_dims=[
                           migraphx.shape.dynamic_dimension(1, 4, {2, 4}),
                           migraphx.shape.dynamic_dimension(3, 3)
                       ])
    m.add_return(
        [m.add_instruction(migraphx.op("neg"), [m.add_parameter("x", s)])])

    scope = {"migraphx": migraphx}
    exec(p.to_py(), scope)
    assert scope["p"].get_parameter_shapes()["x"] == s


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
    test_create_symbolic_shape_from_strings()
    test_create_symbolic_shape_compound_expression()
    test_create_symbolic_shape_with_strides()
    test_create_symbolic_shape_multiple_constraints()
    test_symbol_table_round_trip()
    test_symbol_table_strides_and_multiple_constraints()
    test_symbol_table_empty_without_symbols()
    test_symbol_name_must_be_an_identifier()
    test_to_py_preserves_symbolic_expression()
    test_to_py_preserves_symbolic_strides()
    test_to_py_mixed_symbolic_and_range()
    test_to_py_preserves_dyn_dim_optimals()
    test_parse_onnx_symbolic_dyn_input()

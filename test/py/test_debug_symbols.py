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


def test_module_has_no_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    mm.add_instruction(migraphx.op("add"), [x, y])
    assert not mm.has_debug_symbols()


def test_module_add_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])

    assert not mm.has_debug_symbols()
    mm.add_debug_symbols(add_ins, {"sym_a", "sym_b"})
    assert mm.has_debug_symbols()


def test_module_add_debug_symbols_multiple_instructions():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    relu_ins = mm.add_instruction(migraphx.op("relu"), [add_ins])

    mm.add_debug_symbols(add_ins, {"onnx:add"})
    mm.add_debug_symbols(relu_ins, {"onnx:relu"})

    assert mm.has_debug_symbols()
    assert add_ins.get_debug_symbols() == {"onnx:add"}
    assert relu_ins.get_debug_symbols() == {"onnx:relu"}


def test_module_add_debug_symbols_merge():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])

    mm.add_debug_symbols(add_ins, {"sym_a"})
    mm.add_debug_symbols(add_ins, {"sym_b"})
    assert add_ins.get_debug_symbols() == {"sym_a", "sym_b"}


def test_module_remove_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])

    mm.add_debug_symbols(add_ins, {"sym_a", "sym_b"})
    assert mm.has_debug_symbols()

    mm.remove_debug_symbols(add_ins)
    assert add_ins.get_debug_symbols() == set()
    assert not mm.has_debug_symbols()


def test_module_remove_one_of_two():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    relu_ins = mm.add_instruction(migraphx.op("relu"), [add_ins])

    mm.add_debug_symbols(add_ins, {"sym_add"})
    mm.add_debug_symbols(relu_ins, {"sym_relu"})

    mm.remove_debug_symbols(add_ins)
    assert add_ins.get_debug_symbols() == set()
    assert relu_ins.get_debug_symbols() == {"sym_relu"}
    assert mm.has_debug_symbols()


def test_module_remove_then_readd():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])

    mm.add_debug_symbols(add_ins, {"old_sym"})
    mm.remove_debug_symbols(add_ins)
    assert not mm.has_debug_symbols()

    mm.add_debug_symbols(add_ins, {"new_sym"})
    assert add_ins.get_debug_symbols() == {"new_sym"}
    assert mm.has_debug_symbols()


def test_instruction_get_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])

    assert add_ins.get_debug_symbols() == set()

    mm.add_debug_symbols(add_ins, {"sym_a", "sym_b", "sym_c"})
    assert add_ins.get_debug_symbols() == {"sym_a", "sym_b", "sym_c"}


def test_iterate_instructions_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    z = mm.add_parameter("z", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    mul_ins = mm.add_instruction(migraphx.op("mul"), [add_ins, z])
    relu_ins = mm.add_instruction(migraphx.op("relu"), [mul_ins])
    mm.add_return([relu_ins])

    mm.add_debug_symbols(add_ins, {"onnx:add"})
    mm.add_debug_symbols(mul_ins, {"onnx:mul"})
    mm.add_debug_symbols(relu_ins, {"onnx:relu"})

    all_symbols = set()
    for ins in mm:
        all_symbols.update(ins.get_debug_symbols())

    assert all_symbols == {"onnx:add", "onnx:mul", "onnx:relu"}


def test_iterate_only_symbolized_instructions():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    relu_ins = mm.add_instruction(migraphx.op("relu"), [add_ins])
    mm.add_return([relu_ins])

    mm.add_debug_symbols(relu_ins, {"onnx:relu"})

    symbolized = {ins.name(): ins.get_debug_symbols()
                  for ins in mm if ins.get_debug_symbols()}
    assert len(symbolized) == 1
    assert symbolized["relu"] == {"onnx:relu"}


def test_parse_onnx_with_debug_symbols():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx",
                            use_debug_symbols=True)
    mm = p.get_main_module()
    assert mm.has_debug_symbols()

    all_symbols = set()
    for ins in mm:
        all_symbols.update(ins.get_debug_symbols())
    assert len(all_symbols) > 0


def test_parse_onnx_without_debug_symbols():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx",
                            use_debug_symbols=False)
    mm = p.get_main_module()
    assert not mm.has_debug_symbols()


if __name__ == "__main__":
    test_module_has_no_debug_symbols()
    test_module_add_debug_symbols()
    test_module_add_debug_symbols_multiple_instructions()
    test_module_add_debug_symbols_merge()
    test_module_remove_debug_symbols()
    test_module_remove_one_of_two()
    test_module_remove_then_readd()
    test_instruction_get_debug_symbols()
    test_iterate_instructions_debug_symbols()
    test_iterate_only_symbolized_instructions()
    test_parse_onnx_with_debug_symbols()
    test_parse_onnx_without_debug_symbols()

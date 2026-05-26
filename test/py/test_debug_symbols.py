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


def _any_debug_symbols(mod):
    return any(ins.get_debug_symbols() for ins in mod)


def _make_arg(lens, data):
    return migraphx.create_argument(
        migraphx.shape(type="float_type", lens=lens), data)


def test_no_debug_symbols_by_default():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_ins])
    assert not _any_debug_symbols(mm)
    for ins in mm:
        assert ins.get_debug_symbols() == set()


def test_add_instruction_with_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y],
                                 debug_symbols=["sym_a", "sym_b"])
    assert add_ins.get_debug_symbols() == {"sym_a", "sym_b"}
    assert _any_debug_symbols(mm)


def test_add_instruction_empty_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y], debug_symbols=[])
    assert add_ins.get_debug_symbols() == set()
    assert not _any_debug_symbols(mm)


def test_add_instruction_with_mod_args_and_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y], [],
                                 debug_symbols=["onnx:add"])
    assert add_ins.get_debug_symbols() == {"onnx:add"}


def test_add_literal_argument_with_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(_make_arg([2, 2], [1.0, 2.0, 3.0, 4.0]),
                       debug_symbols=["weights:w0"])
    assert a.get_debug_symbols() == {"weights:w0"}


def test_add_literal_buffer_with_debug_symbols():
    import numpy as np
    p = migraphx.program()
    mm = p.get_main_module()
    buf = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = mm.add_literal(buf, debug_symbols=["weights:w1"])
    assert a.get_debug_symbols() == {"weights:w1"}


def test_add_parameter_with_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s, debug_symbols=["input:x"])
    assert x.get_debug_symbols() == {"input:x"}


def test_add_return_with_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    ret_ins = mm.add_return([add_ins], debug_symbols=["@output_0:result"])
    assert ret_ins.get_debug_symbols() == {"@output_0:result"}
    for ins in mm:
        if ins.name() == "@return":
            assert ins.get_debug_symbols() == {"@output_0:result"}


def test_replace_return_with_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_ins])
    relu_ins = mm.add_instruction(migraphx.op("relu"), [add_ins])
    new_ret = mm.replace_return([relu_ins], debug_symbols=["@output_0:relu"])
    assert new_ret.get_debug_symbols() == {"@output_0:relu"}


def test_add_macro_with_debug_symbols_tags_all_added():
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(_make_arg([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    b = mm.add_literal(_make_arg([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
    # Inputs to the macro should NOT be tagged.
    assert a.get_debug_symbols() == set()
    assert b.get_debug_symbols() == set()

    mac = migraphx.macro("gemm", alpha=2.0)
    result = mm.add_macro(mac, [a, b], debug_symbols=["macro:gemm"])

    # All returned outputs must carry the symbol.
    for ins in result:
        assert "macro:gemm" in ins.get_debug_symbols()

    # The macro added at least one internal mul (alpha-scale) plus a dot, so
    # at least 2 instructions should be tagged.
    tagged = [ins for ins in mm if ins.get_debug_symbols() == {"macro:gemm"}]
    assert len(tagged) >= 2

    # Original inputs (literals) must remain untagged.
    assert a.get_debug_symbols() == set()
    assert b.get_debug_symbols() == set()


def test_add_macro_without_debug_symbols_is_unchanged():
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(_make_arg([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    b = mm.add_literal(_make_arg([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
    mac = migraphx.macro("gemm")
    result = mm.add_macro(mac, [a, b])
    mm.add_return([result[-1]])
    assert not _any_debug_symbols(mm)


def test_insert_macro_with_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(_make_arg([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    b = mm.add_literal(_make_arg([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
    gemm_mac = migraphx.macro("gemm")
    gemm_result = mm.add_macro(gemm_mac, [a, b])
    einsum_mac = migraphx.macro("einsum", equation="ij,jk->ik")
    einsum_result = mm.insert_macro(gemm_result[0], einsum_mac, [a, b],
                                    debug_symbols=["macro:einsum"])

    for ins in einsum_result:
        assert "macro:einsum" in ins.get_debug_symbols()

    # gemm wasn't tagged, so its instructions stay clean.
    for ins in gemm_result:
        assert ins.get_debug_symbols() == set()


def test_iterate_instructions_debug_symbols():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    z = mm.add_parameter("z", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y],
                                 debug_symbols=["onnx:add"])
    mul_ins = mm.add_instruction(migraphx.op("mul"), [add_ins, z],
                                 debug_symbols=["onnx:mul"])
    relu_ins = mm.add_instruction(migraphx.op("relu"), [mul_ins],
                                  debug_symbols=["onnx:relu"])
    mm.add_return([relu_ins])

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
    relu_ins = mm.add_instruction(migraphx.op("relu"), [add_ins],
                                  debug_symbols=["onnx:relu"])
    mm.add_return([relu_ins])

    symbolized = {ins.name(): ins.get_debug_symbols()
                  for ins in mm if ins.get_debug_symbols()}
    assert len(symbolized) == 1
    assert symbolized["relu"] == {"onnx:relu"}


def test_parse_onnx_with_debug_symbols():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx",
                            use_debug_symbols=True)
    mm = p.get_main_module()
    assert _any_debug_symbols(mm)

    all_symbols = set()
    for ins in mm:
        all_symbols.update(ins.get_debug_symbols())
    assert len(all_symbols) > 0


def test_parse_onnx_without_debug_symbols():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx",
                            use_debug_symbols=False)
    mm = p.get_main_module()
    assert not _any_debug_symbols(mm)


if __name__ == "__main__":
    test_no_debug_symbols_by_default()
    test_add_instruction_with_debug_symbols()
    test_add_instruction_empty_debug_symbols()
    test_add_instruction_with_mod_args_and_debug_symbols()
    test_add_literal_argument_with_debug_symbols()
    test_add_literal_buffer_with_debug_symbols()
    test_add_parameter_with_debug_symbols()
    test_add_return_with_debug_symbols()
    test_replace_return_with_debug_symbols()
    test_add_macro_with_debug_symbols_tags_all_added()
    test_add_macro_without_debug_symbols_is_unchanged()
    test_insert_macro_with_debug_symbols()
    test_iterate_instructions_debug_symbols()
    test_iterate_only_symbolized_instructions()
    test_parse_onnx_with_debug_symbols()
    test_parse_onnx_without_debug_symbols()

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
import migraphx, array, math


def create_buffer(t, data, shape):
    a = array.array(t, data)
    m = memoryview(a.tobytes())
    return m.cast(t, shape)


def test_macro_name_and_options():
    mac = migraphx.macro("gemm", alpha=2.0, transB=True)
    assert mac.name() == "gemm"
    opts = mac.options()
    assert opts["alpha"] == 2.0
    assert opts["transB"] == True


def test_macro_no_options():
    mac = migraphx.macro("gemm")
    assert mac.name() == "gemm"
    assert mac.options() == {}


def test_add_macro_gemm():
    p = migraphx.program()
    mm = p.get_main_module()
    # 2x3 matrix
    a = mm.add_literal(
        create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)))
    # 3x2 matrix
    b = mm.add_literal(
        create_buffer('f', [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2)))
    mac = migraphx.macro("gemm")
    result = mm.add_macro(mac, [a, b])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # [1,2,3;4,5,6] . [7,8;9,10;11,12] = [58,64; 139,154]
    assert output == [58.0, 64.0, 139.0, 154.0]


def test_add_macro_gemm_transA():
    p = migraphx.program()
    mm = p.get_main_module()
    # 3x2 matrix, will be transposed to 2x3
    a = mm.add_literal(
        create_buffer('f', [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], (3, 2)))
    # 3x2 matrix
    b = mm.add_literal(
        create_buffer('f', [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2)))
    mac = migraphx.macro("gemm", transA=True)
    result = mm.add_macro(mac, [a, b])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # transposed A = [1,2,3;4,5,6], B = [7,8;9,10;11,12]
    # = [58,64; 139,154]
    assert output == [58.0, 64.0, 139.0, 154.0]


def test_add_macro_gemm_with_params():
    p = migraphx.program()
    mm = p.get_main_module()
    s_a = migraphx.shape(lens=[2, 3], type="float")
    s_b = migraphx.shape(lens=[3, 2], type="float")
    a = mm.add_parameter("a", s_a)
    b = mm.add_parameter("b", s_b)
    mac = migraphx.macro("gemm")
    result = mm.add_macro(mac, [a, b])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    a_data = create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
    b_data = create_buffer('f', [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2))
    output = p.run({"a": a_data, "b": b_data})[-1].tolist()
    assert output == [58.0, 64.0, 139.0, 154.0]


def test_add_macro_batchnorm():
    p = migraphx.program()
    mm = p.get_main_module()
    # Input: (1, 2, 3) - batch=1, channels=2, width=3
    x = mm.add_literal(
        create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3)))
    # scale, bias, mean, var each have shape (2,) for 2 channels
    scale = mm.add_literal(create_buffer('f', [1.0, 1.0], (2,)))
    bias = mm.add_literal(create_buffer('f', [0.0, 0.0], (2,)))
    mean = mm.add_literal(create_buffer('f', [2.0, 5.0], (2,)))
    var = mm.add_literal(create_buffer('f', [1.0, 1.0], (2,)))
    mac = migraphx.macro("batchnorm")
    result = mm.add_macro(mac, [x, scale, bias, mean, var])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # BN: (x - mean) / sqrt(var + eps) * scale + bias
    # Channel 0: (1-2)/sqrt(1+1e-5), (2-2)/sqrt(1+1e-5), (3-2)/sqrt(1+1e-5)
    # Channel 1: (4-5)/sqrt(1+1e-5), (5-5)/sqrt(1+1e-5), (6-5)/sqrt(1+1e-5)
    eps = 1e-5
    expected = [
        (1.0 - 2.0) / math.sqrt(1.0 + eps),
        (2.0 - 2.0) / math.sqrt(1.0 + eps),
        (3.0 - 2.0) / math.sqrt(1.0 + eps),
        (4.0 - 5.0) / math.sqrt(1.0 + eps),
        (5.0 - 5.0) / math.sqrt(1.0 + eps),
        (6.0 - 5.0) / math.sqrt(1.0 + eps),
    ]
    assert len(output) == len(expected)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-4


def test_add_macro_batchnorm_with_epsilon():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(
        create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3)))
    scale = mm.add_literal(create_buffer('f', [2.0, 0.5], (2,)))
    bias = mm.add_literal(create_buffer('f', [1.0, -1.0], (2,)))
    mean = mm.add_literal(create_buffer('f', [2.0, 5.0], (2,)))
    var = mm.add_literal(create_buffer('f', [4.0, 4.0], (2,)))
    mac = migraphx.macro("batchnorm", epsilon=0.01)
    result = mm.add_macro(mac, [x, scale, bias, mean, var])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    eps = 0.01
    expected = [
        2.0 * (1.0 - 2.0) / math.sqrt(4.0 + eps) + 1.0,
        2.0 * (2.0 - 2.0) / math.sqrt(4.0 + eps) + 1.0,
        2.0 * (3.0 - 2.0) / math.sqrt(4.0 + eps) + 1.0,
        0.5 * (4.0 - 5.0) / math.sqrt(4.0 + eps) + (-1.0),
        0.5 * (5.0 - 5.0) / math.sqrt(4.0 + eps) + (-1.0),
        0.5 * (6.0 - 5.0) / math.sqrt(4.0 + eps) + (-1.0),
    ]
    assert len(output) == len(expected)
    for o, e in zip(output, expected):
        assert abs(o - e) < 1e-4


def test_add_macro_convolution():
    p = migraphx.program()
    mm = p.get_main_module()
    # Input: (1, 1, 3, 3) - batch=1, channels=1, 3x3
    x = mm.add_literal(
        create_buffer('f', [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ], (1, 1, 3, 3)))
    # Weight: (1, 1, 2, 2) - out_channels=1, in_channels=1, 2x2 kernel
    w = mm.add_literal(
        create_buffer('f', [1.0, 0.0, 0.0, 1.0], (1, 1, 2, 2)))
    mac = migraphx.macro("convolution")
    result = mm.add_macro(mac, [x, w])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # 2x2 kernel [[1,0],[0,1]] on 3x3 input, stride=1, no padding
    # Output is 2x2:
    # [1*1+2*0+4*0+5*1, 2*1+3*0+5*0+6*1] = [6, 8]
    # [4*1+5*0+7*0+8*1, 5*1+6*0+8*0+9*1] = [12, 14]
    assert output == [6.0, 8.0, 12.0, 14.0]


def test_add_macro_convolution_with_bias():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(
        create_buffer('f', [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ], (1, 1, 3, 3)))
    w = mm.add_literal(
        create_buffer('f', [1.0, 0.0, 0.0, 1.0], (1, 1, 2, 2)))
    bias = mm.add_literal(create_buffer('f', [10.0], (1,)))
    mac = migraphx.macro("convolution")
    result = mm.add_macro(mac, [x, w, bias])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # Same as above + bias of 10
    assert output == [16.0, 18.0, 22.0, 24.0]


def test_add_macro_convolution_with_padding():
    p = migraphx.program()
    mm = p.get_main_module()
    # Input: (1, 1, 3, 3)
    x = mm.add_literal(
        create_buffer('f', [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ], (1, 1, 3, 3)))
    # Weight: (1, 1, 3, 3) - 3x3 kernel all ones
    w = mm.add_literal(
        create_buffer('f', [1.0] * 9, (1, 1, 3, 3)))
    mac = migraphx.macro("convolution", paddings=[1, 1])
    result = mm.add_macro(mac, [x, w])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # 3x3 kernel of ones on 3x3 input with padding=1 => 3x3 output
    # Each output is the sum of the overlapping region
    assert output == [12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0]


def test_add_macro_einsum_matmul():
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(
        create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)))
    b = mm.add_literal(
        create_buffer('f', [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2)))
    mac = migraphx.macro("einsum", equation="ij,jk->ik")
    result = mm.add_macro(mac, [a, b])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # Same as matrix multiply: [58, 64, 139, 154]
    assert output == [58.0, 64.0, 139.0, 154.0]


def test_add_macro_einsum_transpose():
    p = migraphx.program()
    mm = p.get_main_module()
    a = mm.add_literal(
        create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)))
    mac = migraphx.macro("einsum", equation="ij->ji")
    result = mm.add_macro(mac, [a])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # Transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
    assert output == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_add_macro_einsum_trace():
    p = migraphx.program()
    mm = p.get_main_module()
    # 3x3 identity matrix
    a = mm.add_literal(
        create_buffer('f', [
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0
        ], (3, 3)))
    mac = migraphx.macro("einsum", equation="ii->")
    result = mm.add_macro(mac, [a])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    # Trace: 1 + 2 + 3 = 6
    assert output == [6.0]


def test_insert_macro():
    p = migraphx.program()
    mm = p.get_main_module()
    # 2x3 matrix
    a = mm.add_literal(
        create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)))
    # 3x2 matrix
    b = mm.add_literal(
        create_buffer('f', [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2)))
    # First add a gemm at the end
    gemm_mac = migraphx.macro("gemm")
    gemm_result = mm.add_macro(gemm_mac, [a, b])
    # Insert an einsum matmul before the gemm
    einsum_mac = migraphx.macro("einsum", equation="ij,jk->ik")
    einsum_result = mm.insert_macro(gemm_result[0], einsum_mac, [a, b])
    # Return the einsum result
    mm.add_return([einsum_result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    assert output == [58.0, 64.0, 139.0, 154.0]


if __name__ == "__main__":
    test_macro_name_and_options()
    test_macro_no_options()
    test_add_macro_gemm()
    test_add_macro_gemm_transA()
    test_add_macro_gemm_with_params()
    test_add_macro_batchnorm()
    test_add_macro_batchnorm_with_epsilon()
    test_add_macro_convolution()
    test_add_macro_convolution_with_bias()
    test_add_macro_convolution_with_padding()
    test_add_macro_einsum_matmul()
    test_add_macro_einsum_transpose()
    test_add_macro_einsum_trace()
    test_insert_macro()

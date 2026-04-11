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
import migraphx, array


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
    mac = migraphx.macro("add")
    assert mac.name() == "add"
    assert mac.options() == {}


def test_add_macro_add():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(create_buffer('f', [1.0, 2.0, 3.0], (1, 3)))
    y = mm.add_literal(create_buffer('f', [4.0, 5.0, 6.0], (1, 3)))
    mac = migraphx.macro("add")
    result = mm.add_macro(mac, [x, y])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    assert output == [5.0, 7.0, 9.0]


def test_add_macro_mul():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(create_buffer('f', [2.0, 3.0, 4.0], (1, 3)))
    y = mm.add_literal(create_buffer('f', [5.0, 6.0, 7.0], (1, 3)))
    mac = migraphx.macro("mul")
    result = mm.add_macro(mac, [x, y])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    assert output == [10.0, 18.0, 28.0]


def test_add_macro_broadcast():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)))
    y = mm.add_literal(create_buffer('f', [10.0, 20.0, 30.0], (3,)))
    mac = migraphx.macro("add")
    result = mm.add_macro(mac, [x, y])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    assert output == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]


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
    # [1,2,3].[7,8;9,10;11,12] = [58,64; 139,154]
    assert output == [58.0, 64.0, 139.0, 154.0]


def test_add_macro_gemm_with_options():
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


def test_insert_macro():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(create_buffer('f', [1.0, 2.0, 3.0], (1, 3)))
    y = mm.add_literal(create_buffer('f', [4.0, 5.0, 6.0], (1, 3)))
    # First add a mul at the end
    mul_mac = migraphx.macro("mul")
    mul_result = mm.add_macro(mul_mac, [x, y])
    # Insert an add before the mul
    add_mac = migraphx.macro("add")
    add_result = mm.insert_macro(mul_result[0], add_mac, [x, y])
    # Return the add result
    mm.add_return([add_result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    assert output == [5.0, 7.0, 9.0]


def test_add_macro_sub():
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(create_buffer('f', [10.0, 20.0, 30.0], (1, 3)))
    y = mm.add_literal(create_buffer('f', [1.0, 2.0, 3.0], (1, 3)))
    mac = migraphx.macro("sub")
    result = mm.add_macro(mac, [x, y])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    output = p.run({})[-1].tolist()
    assert output == [9.0, 18.0, 27.0]


def test_add_macro_with_params():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    mac = migraphx.macro("add")
    result = mm.add_macro(mac, [x, y])
    mm.add_return([result[-1]])
    p.compile(migraphx.get_target("ref"))
    x_data = create_buffer('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
    y_data = create_buffer('f', [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (2, 3))
    output = p.run({"x": x_data, "y": y_data})[-1].tolist()
    assert output == [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]


if __name__ == "__main__":
    test_macro_name_and_options()
    test_macro_no_options()
    test_add_macro_add()
    test_add_macro_mul()
    test_add_macro_broadcast()
    test_add_macro_gemm()
    test_add_macro_gemm_with_options()
    test_insert_macro()
    test_add_macro_sub()
    test_add_macro_with_params()

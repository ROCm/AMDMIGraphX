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
import migraphx, tempfile, os


def test_netron_output_parsed_model():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as t:
        filename = t.name

    migraphx.save(p, filename, format="onnx_for_netron")
    size = os.path.getsize(filename)
    assert size > 0, "Netron output file is empty"
    os.remove(filename)


def test_netron_output_constructed_program():
    p = migraphx.program()
    mm = p.get_main_module()
    s = migraphx.shape(lens=[2, 3], type="float")
    x = mm.add_parameter("x", s)
    y = mm.add_parameter("y", s)
    add_ins = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_ins])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as t:
        filename = t.name

    migraphx.save(p, filename, format="onnx_for_netron")
    size = os.path.getsize(filename)
    assert size > 0, "Netron output file is empty"
    os.remove(filename)


if __name__ == "__main__":
    test_netron_output_parsed_model()
    test_netron_output_constructed_program()

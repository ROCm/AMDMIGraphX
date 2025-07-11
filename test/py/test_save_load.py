#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
import migraphx, array, tempfile, sys


def test_conv_relu(format):
    p1 = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    print(p1)

    s1 = p1.get_output_shapes()[-1]

    with tempfile.NamedTemporaryFile() as t:
        migraphx.save(p1, t.name, format=format)

        p2 = migraphx.load(t.name, format=format)
        print(p2)
        s2 = p2.get_output_shapes()[-1]

        assert s1 == s2
        assert p1.sort() == p2.sort()


def create_buffer(t, data, shape):
    a = array.array(t, data)
    if sys.version_info >= (3, 0):
        m = memoryview(a.tobytes())
        return m.cast(t, shape)
    else:
        m = memoryview(a.tostring())
        return m

def test_load_save_arg():
    data = [1,2,3,4]
    buffer1 = create_buffer('f', data, [2,2])
    arg1 = migraphx.argument(buffer1)
    migraphx.argument.save(arg1, 'load_save_arg.msgpack')
    arg2 = migraphx.argument.load('load_save_arg.msgpack')
    assert arg1 == arg2

if __name__ == "__main__":
    test_load_save_arg()
    test_conv_relu('msgpack')
    test_conv_relu('json')

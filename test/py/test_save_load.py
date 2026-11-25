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
import migraphx, array, json, os, tempfile, sys


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


def test_external_weights(format):
    p1 = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")

    with tempfile.TemporaryDirectory() as d:
        name = os.path.join(d, "model.mxr")
        migraphx.save(p1, name, format=format, external_weights=1)
        # Each save writes a new generation named model.mxr.<id>.data
        generations = [
            f for f in os.listdir(d)
            if f.startswith("model.mxr.") and f.endswith(".data")
        ]
        assert len(generations) == 1
        assert not os.path.exists(name + ".data")

        p2 = migraphx.load(name, format=format, verify_external_weights=True)
        assert p1.sort() == p2.sort()
        del p2


def external_refs(v):
    if isinstance(v, dict):
        refs = [v["external"]] if "external" in v else []
        return refs + [r for x in v.values() for r in external_refs(x)]
    if isinstance(v, list):
        return [r for x in v for r in external_refs(x)]
    return []


def test_reuse_onnx_external_data(p1):
    ext_dir = os.path.abspath("ext_path")
    with tempfile.TemporaryDirectory() as d:
        name = os.path.join(d, "model.json")
        migraphx.save(p1, name, format="json", reuse_external_data=True)
        assert os.listdir(d) == ["model.json"]

        with open(name) as f:
            refs = external_refs(json.load(f))
        assert refs
        assert all(
            os.path.basename(r["file_path"]) == "conv.weight" for r in refs)

        # The ONNX data lies outside the program's directory
        try:
            migraphx.load(name, format="json")
            assert False, "load outside the allowed directories succeeded"
        except RuntimeError:
            pass
        p2 = migraphx.load(name,
                           format="json",
                           external_data_allowed_paths=[ext_dir])
        assert p1.sort() == p2.sort()
        del p2


def test_save_load_buffer():
    p1 = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    print(p1)

    s1 = p1.get_output_shapes()[-1]

    p1_bytes = migraphx.save_buffer(p1)
    assert isinstance(p1_bytes, bytes)

    p2 = migraphx.load_buffer(p1_bytes)
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
    data = [1, 2, 3, 4]
    buffer1 = create_buffer('f', data, [2, 2])
    arg1 = migraphx.argument(buffer1)
    migraphx.argument.save(arg1, 'load_save_arg.msgpack')
    arg2 = migraphx.argument.load('load_save_arg.msgpack')
    assert arg1 == arg2


if __name__ == "__main__":
    test_load_save_arg()
    test_conv_relu('msgpack')
    test_conv_relu('json')
    test_external_weights('msgpack')
    test_external_weights('json')
    test_reuse_onnx_external_data(
        migraphx.parse_onnx("ext_path/external_data_test.onnx",
                            reuse_external_data=True))
    with open("ext_path/external_data_test.onnx", "rb") as f:
        test_reuse_onnx_external_data(
            migraphx.parse_onnx_buffer(f.read(),
                                       external_data_path="ext_path",
                                       reuse_external_data=True))
    test_save_load_buffer()

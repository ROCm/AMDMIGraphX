import migraphx, tempfile


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


test_conv_relu('msgpack')
test_conv_relu('json')

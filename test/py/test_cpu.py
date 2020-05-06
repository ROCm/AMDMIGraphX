import migraphx, array, sys


def test_conv_relu():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("cpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2
    params = {}

    for key, value in p.get_parameter_shapes().items():
        print("Parameter {} -> {}".format(key, value))
        params[key] = migraphx.generate_argument(value)

    r = p.run(params)[-1]
    print(r)


def create_buffer(t, data, shape):
    a = array.array(t, data)
    if sys.version_info >= (3, 0):
        m = memoryview(a.tobytes())
        return m.cast(t, shape)
    else:
        m = memoryview(a.tostring())
        return m


def test_add_scalar():
    p = migraphx.parse_onnx("add_scalar_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("cpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2

    d0 = list(range(120))
    arg0 = create_buffer("B", d0, [2, 3, 4, 5])
    d1 = [1]
    arg1 = create_buffer("B", d1, ())

    params = {}
    params["0"] = migraphx.argument(arg0)
    params["1"] = migraphx.argument(arg1)

    r = p.run(params)[-1]
    print(r)


test_conv_relu()
if sys.version_info >= (3, 0):
    test_add_scalar()

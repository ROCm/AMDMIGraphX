import migraphx
try:
    import numpy as np
except:
    pass


def test_conv_relu():
    p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    for key, value in p.get_parameter_shapes().items():
        print("Parameter {} -> {}".format(key, value))
        params[key] = migraphx.generate_argument(value)

    r = p.run(params)
    print(r)


def test_sub_uint64():
    p = migraphx.parse_onnx("implicit_sub_bcast_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    shapes = p.get_parameter_shapes()
    arg0 = np.arange(120).reshape(shapes["0"].lens()).astype(np.uint64)
    arg1 = np.arange(20).reshape(shapes["1"].lens()).astype(np.uint64)

    params["0"] = migraphx.argument(arg0)
    params["1"] = migraphx.argument(arg1)
    r = p.run(params)
    print(r)


def test_neg_int64():
    p = migraphx.parse_onnx("neg_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    shapes = p.get_parameter_shapes()
    arg0 = np.arange(6).reshape(shapes["0"].lens()).astype(np.int64)

    params["0"] = migraphx.argument(arg0)
    r = p.run(params)
    print(r)


def test_fp16_imagescaler():
    p = migraphx.parse_onnx("imagescaler_half_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2

    params = {}
    shapes = p.get_parameter_shapes()
    arg0 = np.random.randn(768).reshape(shapes["0"].lens()).astype(np.float16)
    params["0"] = migraphx.argument(arg0)

    r = p.run(params)[-1]
    print(r)


test_conv_relu()
test_sub_uint64()
test_neg_int64()
test_fp16_imagescaler()

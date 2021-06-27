import sys
import migraphx
try:
    import numpy as np
except:
    sys.exit()


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
    params["0"] = np.arange(120).reshape(shapes["0"].lens()).astype(np.uint64)
    params["1"] = np.arange(20).reshape(shapes["1"].lens()).astype(np.uint64)

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
    params["0"] = np.arange(6).reshape(shapes["0"].lens()).astype(np.int64)

    r = p.run(params)
    print(r)


def test_nonzero_dynamic():
    p = migraphx.parse_onnx("nonzero_dynamic_test.onnx")
    print(p)
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    params = {}

    shapes = p.get_parameter_shapes()
    params["data"] = np.array([1, 1, 0, 1]).reshape(
        shapes["data"].lens()).astype(np.bool)

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
    params["0"] = np.random.randn(768).reshape(shapes["0"].lens()).astype(
        np.float16)

    r = p.run(params)[-1]
    print(r)


def test_if_pl():
    p = migraphx.parse_onnx("if_pl_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("gpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2

    params = {}
    shapes = p.get_parameter_shapes()
    params["x"] = np.ones(6).reshape(shapes["x"].lens()).astype(np.float32)
    params["y"] = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
                            ]).reshape(shapes["y"].lens()).astype(np.float32)
    params["cond"] = np.array([1]).reshape(()).astype(np.bool)

    r = p.run(params)[-1]
    print(r)


test_conv_relu()
test_sub_uint64()
test_neg_int64()
test_fp16_imagescaler()
test_if_pl()
test_nonzero_dynamic()

import migraphx


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
        print("Parameter {} -> {}".format(key, value))0
        params[key] = migraphx.generate_argument(value)

    r = p.run(params)[-1]
    print(r)


def test_add_scalar():
    p = migraphx.parse_onnx("add_scalar_test.onnx")
    print(p)
    s1 = p.get_output_shapes()[-1]
    print("Compiling ...")
    p.compile(migraphx.get_target("cpu"))
    print(p)
    s2 = p.get_output_shapes()[-1]
    assert s1 == s2
    params = {}

    for key, value in p.get_parameter_shapes().items():
        print("Parameter {} -> {}".format(key, value))0
        params[key] = migraphx.generate_argument(value)

    # args = []
    # args.append(np.random.randn(2, 3, 4, 5).astype(np.single))
    # args.append(np.array(1).astype(np.single))
    # params["0"] = migraphx.argument(args[0])
    # params["1"] = migraphx.argument(args[1])

    r = p.run(params)[-1]
    print(r)


test_conv_relu()
test_add_scalar()

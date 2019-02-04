import migraphx

p = migraphx.parse_onnx("conv_relu_maxpool.onnx")
print(p)
print("Compiling ...")
p.compile(migraphx.get_target("cpu"))
print(p)
params = {}
for key, value in p.get_parameter_shapes().items():
    print("Parameter {} -> {}".format(key, value))
    params[key] = migraphx.generate_argument(value)

r = p.run(params)
print(r)

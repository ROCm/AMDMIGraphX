import migraphx

p = migraphx.parse_onnx("conv_relu_maxpool.onnx")
p.compile(migraphx.get_target("cpu"))

params = {}
for key, value in p.get_parameter_shapes().items():
    params[key] = migraphx.generate_argument(value)

r = p.run(params)
print(r)

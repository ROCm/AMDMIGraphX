import migraphx

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

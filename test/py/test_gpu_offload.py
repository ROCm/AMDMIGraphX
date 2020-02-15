import migraphx

p = migraphx.parse_onnx("conv_relu_maxpool_test.onnx")
print(p)
print("Compiling ...")
p.compile(migraphx.get_target("gpu"), offload_copy=False)
print(p)
params = {}

for key, value in p.get_parameter_shapes().items():
    print("Parameter {} -> {}".format(key, value))
    params[key] = migraphx.to_gpu(migraphx.generate_argument(value))

r = migraphx.from_gpu(p.run(params)[-1])
print(r)

import migraphx

p = migraphx.parse_onnx("conv_relu_maxpool.onnx")
p.compile(migraphx.get_target("gpu"))
params = {}
for key, value in p.get_parameter_shapes().items():
    params[key] = migraphx.to_gpu(migraphx.generate_argument(value))

r1 = migraphx.from_gpu(p.run(params))
r2 = migraphx.from_gpu(p.run(params))

assert r1 == r2
q1 = memoryview(r1)
q2 = memoryview(r2)
assert q1.tobytes() == q2.tobytes()

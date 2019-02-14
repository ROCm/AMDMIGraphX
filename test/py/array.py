import migraphx, struct

def assert_eq(x, y):
    if x == y:
        pass
    else:
        raise Exception(str(x) + " != " + str(y))

def get_lens(m):
    return list(m.shape)

def get_strides(m):
    return [s/m.itemsize for s in m.strides]

def read_float(b, index):
    return struct.unpack_from('f', b, index*4)[0]

def check_list(a, b):
    l = a.tolist()
    for i in range(len(l)):
        assert_eq(l[i], read_float(b, i))

def run(p):
    params = {}
    for key, value in p.get_parameter_shapes().items():
        params[key] = migraphx.to_gpu(migraphx.generate_argument(value))

    return migraphx.from_gpu(p.run(params))


p = migraphx.parse_onnx("conv_relu_maxpool.onnx")
p.compile(migraphx.get_target("gpu"))

r1 = run(p)
r2 = run(p)
assert_eq(r1, r2)
assert_eq(r1.tolist(), r2.tolist())

assert_eq(r1.tolist()[0], read_float(r1, 0))

m1 = memoryview(r1)
m2 = memoryview(r2)

assert_eq(r1.get_shape().elements(), reduce(lambda x,y: x*y,get_lens(m1), 1))
assert_eq(r1.get_shape().lens(), get_lens(m1))
assert_eq(r1.get_shape().strides(), get_strides(m1))

check_list(r1, m1.tobytes())
check_list(r2, m2.tobytes())

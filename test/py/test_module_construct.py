import migraphx, array, sys
try:
    import numpy as np
except:
    sys.exit()


def create_buffer(t, data, shape):
    a = array.array(t, data)
    if sys.version_info >= (3, 0):
        m = memoryview(a.tobytes())
        return m.cast(t, shape)
    else:
        m = memoryview(a.tostring())
        return m


def test_add_op():
    p = migraphx.program()
    mm = p.get_main_module()
    param_shape = migraphx.shape(lens=[3, 3], type="float")
    # test add_literal with python array
    x = mm.add_literal(param_shape, create_buffer('f', [1.0] * 9, (3, 3)))
    # test add_literal with numpy array
    y = mm.add_literal(param_shape, np.ones((3, 3), dtype='float32'))
    add_op = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_op])
    p.compile(migraphx.get_target("ref"))
    params = {}
    output = p.run(params)[-1].tolist()
    assert output == list(2 * np.ones((9), dtype='float32'))


def test_if_then_else():
    param_shape = migraphx.shape(lens=[3, 3], type="float")
    cond_shape = migraphx.shape(type="bool", lens=[1], strides=[0])

    def create_program():
        p = migraphx.program()
        mm = p.get_main_module()
        cond = mm.add_parameter("cond", cond_shape)
        x = mm.add_parameter("x", param_shape)
        y = mm.add_parameter("y", param_shape)
        then_mod = p.create_module("If_0_if")
        x_identity = then_mod.add_instruction(migraphx.op("identity"), [x])
        then_mod.add_return([x_identity])

        else_mod = p.create_module("If_0_else")
        y_identity = else_mod.add_instruction(migraphx.op("identity"), [y])
        else_mod.add_return([y_identity])

        if_ins = mm.add_instruction(migraphx.op("if"), [cond],
                                    [then_mod, else_mod])
        ret = mm.add_instruction(migraphx.op("get_tuple_elem", **{"index": 0}),
                                 [if_ins])
        mm.add_return([ret])
        return p

    params = {}
    params["x"] = migraphx.generate_argument(param_shape)
    params["y"] = migraphx.generate_argument(param_shape)

    def run_prog(cond):
        p = create_program()
        p.compile(migraphx.get_target("ref"))
        params["cond"] = migraphx.fill_argument(cond_shape, cond)
        output = p.run(params)[-1]
        return output

    assert run_prog(True) == params["x"]
    assert run_prog(False) == params["y"]


if __name__ == "__main__":
    test_add_op()
    test_if_then_else()

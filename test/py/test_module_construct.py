import migraphx
import numpy as np

def test_add_op():
    p = migraphx.program()
    mm = p.get_main_module()
    param_shape = migraphx.shape(lens=[3, 3], type="float")
    x = mm.add_parameter("x", param_shape)
    y = mm.add_parameter("y", param_shape)
    add_op = mm.add_instruction(migraphx.op("add"), [x, y])
    r = mm.add_return([add_op])
    p.compile(migraphx.get_target("ref"))
    params = {}
    x_arg = np.arange(9).reshape(param_shape.lens()).astype(np.float32)
    params["x"] =     params["y"] = np.arange(9).reshape(param_shape.lens()).astype(np.float32)
    output = p.run(params)[-1]
    assert(np.array_equal(output, params["x"] + params["y"]))
    
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

        if_ins = mm.add_instruction(migraphx.op("if"), [cond], [then_mod, else_mod])
        ret = mm.add_instruction(migraphx.op("get_tuple_elem", **{"index": 0}), [if_ins])
        mm.add_return([ret])
        return p

    params = {}
    params["x"] = np.arange(9).reshape(param_shape.lens()).astype(np.float32)
    params["y"] = 2 * np.arange(9).reshape(param_shape.lens()).astype(np.float32)

    def run_prog(cond):
        p = create_program()
        p.compile(migraphx.get_target("ref"))
        params["cond"] = np.array([cond]).reshape(()).astype(np.bool)
        output = p.run(params)[-1]
        return output  

    assert(np.array_equal(run_prog(1), params["x"]))
    assert(np.array_equal(run_prog(0), params["y"]))


if __name__ == "__main__":
    test_add_op()
    test_if_then_else()
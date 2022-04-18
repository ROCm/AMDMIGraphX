import migraphx, sys, array
try:
    import numpy as np
except:
    sys.exit()

def test_add_op():
    p = migraphx.program()
    mm = p.get_main_module()
    param_shape = migraphx.shape(lens=[3, 3], type="float")
    x = mm.add_literal(np.ones((3, 3), dtype='float32'))
    y = mm.add_literal(2 * np.ones((3, 3), dtype='float32'))
    add_op = mm.add_instruction(migraphx.op("add"), [x, y])
    mm.add_return([add_op])
    p.compile(migraphx.get_target("ref"))
    params = {}
    output = p.run(params)[-1].tolist()
    assert output == list(3 * np.ones((9), dtype='float32'))

if __name__ == "__main__":
    test_add_op()

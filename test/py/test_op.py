import migraphx


def test_add_op():
    add_op = migraphx.op("add")
    name = add_op.name()

    assert name == "add"


def test_reduce_mean():
    reduce_mean_op = migraphx.op("reduce_mean", **{"axes": [1, 2, 3, 4]})
    name = reduce_mean_op.name()

    assert name == "reduce_mean"


test_add_op()
test_reduce_mean()

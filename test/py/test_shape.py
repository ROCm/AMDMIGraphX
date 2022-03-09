import migraphx


def test_create_shape():
    s = migraphx.shape(lens=[1, 64, 3, 3])
    assert s.standard()
    assert s.packed()
    assert s.lens() == [1, 64, 3, 3]


def test_create_shape_broadcast():
    s = migraphx.shape(lens=[1, 64, 3, 3], strides=[0, 1, 0, 0])
    assert s.broadcasted()
    assert s.lens() == [1, 64, 3, 3]
    assert s.strides() == [0, 1, 0, 0]


def test_create_shape_type():
    s = migraphx.shape(type='uint8')
    assert s.type_string() == 'uint8_type'
    assert s.type_size() == 1

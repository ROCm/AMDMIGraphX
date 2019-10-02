import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto


def onnx_test(op_test):
    def run_test():
        op_info = op_test()
        if len(op_info) > 3:
            graph_def = helper.make_graph(op_info[0],
                                          op_test.__name__,
                                          op_info[1],
                                          op_info[2],
                                          initializer=op_info[3])
        else:
            graph_def = helper.make_graph(op_info[0], op_test.__name__,
                                          op_info[1], op_info[2])
        model_def = helper.make_model(graph_def,
                                      producer_name=op_test.__name__)
        onnx.save(model_def, '{}.onnx'.format(op_test.__name__))

    return run_test


@onnx_test
def acos_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Acos',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def add_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Add',
                                 inputs=['0', '1'],
                                 broadcast=1,
                                 axis=1,
                                 outputs=['2'])

    return ([node], [x, y], [z])


@onnx_test
def add_fp16_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [1])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [1])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return (
        [node],
        [x, y],
        [z],
        # '0' -> 1.5, '1' -> 2.5
        [
            onnx.helper.make_tensor('0', TensorProto.FLOAT16, [1], [15872]),
            onnx.helper.make_tensor('1', TensorProto.FLOAT16, [1], [16640])
        ])


@onnx_test
def add_scalar_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Add', inputs=['0', '1'], outputs=['2'])

    return ([node], [x, y], [z],
            [helper.make_tensor('1', TensorProto.FLOAT, [], [1])])


@onnx_test
def argmax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])

    node = onnx.helper.make_node('ArgMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=2,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def argmin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])

    node = onnx.helper.make_node('ArgMin',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=3,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def asin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Asin',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def atan_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Atan',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def cast_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node('Cast', inputs=['x'], outputs=['y'], to=1)

    return ([node], [x], [y])


@onnx_test
def ceil_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Ceil',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def clip_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Clip',
                                 inputs=['0'],
                                 outputs=['1'],
                                 max=6.0,
                                 min=0.0)

    return ([node], [x], [y])


@onnx_test
def concat_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 4, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7, 4, 3])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [9, 4, 3])

    node = onnx.helper.make_node(
        'Concat',
        inputs=['0', '1'],
        axis=0,
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test
def constant_test():
    x = np.array([0, 1, 2])
    y = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['0'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        ),
    )

    return ([node], [], [y])


@onnx_test
def constant_fill_test():
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'ConstantFill',
        inputs=[],
        outputs=['value'],
        dtype=1,
        value=1.0,
        shape=[2, 3],
        input_as_shape=0,
    )

    return ([node], [], [value])


@onnx_test
def constant_fill_input_as_shape_test():
    np_shape = np.array([2, 3])
    shape = helper.make_tensor_value_info('shape', TensorProto.INT32, [2])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [2, 3])

    ts_shape = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=np_shape.shape,
                                  vals=np_shape.flatten().astype(int))

    const_shape_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=ts_shape,
    )

    node = onnx.helper.make_node(
        'ConstantFill',
        inputs=['shape'],
        outputs=['value'],
        dtype=1,
        value=1.0,
        input_as_shape=1,
    )

    return ([const_shape_node, node], [], [value])


@onnx_test
def constant_scalar_test():
    x = np.array([1])
    y = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['0'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.INT32,
            dims=x.shape,
            vals=x.flatten().astype(int),
        ),
    )

    return ([node], [], [y])


@onnx_test
def const_of_shape_empty_input_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.INT64, [1],
                                         [10])
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    empty_val = np.array([]).astype(np.int64)
    empty_ts = helper.make_tensor(name='empty_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=empty_val.shape,
                                  vals=empty_val.flatten().astype(int))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=empty_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['shape'],
        outputs=['y'],
        value=tensor_val,
    )

    return ([shape_const, node], [], [y])


@onnx_test
def const_of_shape_float_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1],
                                         [10])

    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(int))

    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['shape'],
                                 outputs=['y'],
                                 value=tensor_val)

    return ([shape_const, node], [], [y])


@onnx_test
def const_of_shape_int64_test():
    tensor_val = onnx.helper.make_tensor('value', onnx.TensorProto.INT64, [1],
                                         [10])
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(int))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node('ConstantOfShape',
                                 inputs=['shape'],
                                 outputs=['y'],
                                 value=tensor_val)

    return ([shape_const, node], [], [y])


@onnx_test
def const_of_shape_no_value_attr_test():
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(int))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])

    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['shape'],
        outputs=['y'],
    )

    return ([shape_const, node], [], [y])


@onnx_test
def conv_autopad_fail_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])
    out = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 34, 34])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 dilations=[1, 1],
                                 strides=[1, 1],
                                 auto_pad='SAME',
                                 pads=[0, 0, 1, 1, 0, 0, 1, 1])

    return ([node], [x, y], [out])


@onnx_test
def conv_bias_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 2, 28, 28])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'],
                                 dilations=[1, 1],
                                 strides=[1, 1])

    return ([node], [x, y, z], [out])


@onnx_test
def conv_bn_relu_maxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    m = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1])
    n = helper.make_tensor_value_info('4', TensorProto.FLOAT, [1])
    k = helper.make_tensor_value_info('5', TensorProto.FLOAT, [1])
    l = helper.make_tensor_value_info('6', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('10', TensorProto.FLOAT,
                                        [1, 1, 14, 14])

    node0 = onnx.helper.make_node('Conv',
                                  inputs=['0', '1', '2'],
                                  outputs=['7'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node1 = onnx.helper.make_node('BatchNormalization',
                                  inputs=['7', '3', '4', '5', '6'],
                                  outputs=['8'],
                                  epsilon=9.99999974737875e-06,
                                  momentum=0.899999976158142)

    node2 = onnx.helper.make_node('Relu', inputs=['8'], outputs=['9'])
    node3 = onnx.helper.make_node('MaxPool',
                                  inputs=['9'],
                                  outputs=['10'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    return ([node0, node1, node2, node3], [x, y, z, m, n, k, l], [out])


@onnx_test
def conv_relu_maxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('5', TensorProto.FLOAT, [1, 1, 14, 14])

    node1 = onnx.helper.make_node('Conv',
                                  inputs=['0', '1', '2'],
                                  outputs=['3'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node2 = onnx.helper.make_node('Relu', inputs=['3'], outputs=['4'])

    node3 = onnx.helper.make_node('MaxPool',
                                  inputs=['4'],
                                  outputs=['5'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    return ([node1, node2, node3], [x, y, z], [out])


@onnx_test
def conv_relu_maxpool_x2_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [5, 3, 5, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5])
    m = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 5, 5, 5])
    n = helper.make_tensor_value_info('4', TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info('10', TensorProto.FLOAT, [1, 1, 5, 5])

    node1 = onnx.helper.make_node('Conv',
                                  inputs=['0', '1', '2'],
                                  outputs=['5'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node2 = onnx.helper.make_node('Relu', inputs=['5'], outputs=['6'])

    node3 = onnx.helper.make_node('MaxPool',
                                  inputs=['6'],
                                  outputs=['7'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    node4 = onnx.helper.make_node('Conv',
                                  inputs=['7', '3', '4'],
                                  outputs=['8'],
                                  dilations=[1, 1],
                                  strides=[1, 1],
                                  pads=[0, 0, 0, 0])

    node5 = onnx.helper.make_node('Relu', inputs=['8'], outputs=['9'])

    node6 = onnx.helper.make_node('MaxPool',
                                  inputs=['9'],
                                  outputs=['10'],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2],
                                  kernel_shape=[2, 2])

    return ([node1, node2, node3, node4, node5, node6], [x, y, z, m, n], [out])


@onnx_test
def cos_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Cos',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def cosh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Cosh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def dropout_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

    node = onnx.helper.make_node(
        'Dropout',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test
def elu_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('Elu',
                                 inputs=['0'],
                                 outputs=['1'],
                                 alpha=0.01)

    return ([node], [x], [y])


@onnx_test
def erf_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 15])

    node = onnx.helper.make_node(
        'Erf',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def exp_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Exp',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def expand_test():
    shape_val = np.array([2, 3, 4, 5]).astype(np.int64)
    shape_ts = helper.make_tensor(name='shape_tensor',
                                  data_type=TensorProto.INT32,
                                  dims=shape_val.shape,
                                  vals=shape_val.flatten().astype(int))
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Expand',
                                 inputs=['x', 'shape'],
                                 outputs=['y'])

    return ([shape_const, node], [x], [y])


@onnx_test
def flatten_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [6, 20])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 60])

    node = onnx.helper.make_node('Flatten',
                                 inputs=['0'],
                                 axis=2,
                                 outputs=['2'])

    node2 = onnx.helper.make_node('Flatten', inputs=['0'], outputs=['3'])

    return ([node, node2], [x], [y, y2])


@onnx_test
def floor_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Floor',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def gather_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32,
                                      [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=1,
    )

    return ([node], [x, i], [y])


@onnx_test
def gemm_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5, 7])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [11, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [])
    a = helper.make_tensor_value_info('3', TensorProto.FLOAT, [7, 11])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['0', '1', '2'],
                                 outputs=['3'],
                                 alpha=2.0,
                                 beta=2.0,
                                 transA=1,
                                 transB=1)

    return ([node], [x, y, z], [a])


@onnx_test
def gemm_ex_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 1, 5, 6])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 5, 7])
    m3 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 1, 6, 7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['1', '2', '3'],
                                 outputs=['y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [m1, m2, m3], [y])


@onnx_test
def gemm_ex_brcst_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 1, 5, 6])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 5, 7])
    m3 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 1, 6, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 6, 7])

    node = onnx.helper.make_node('Gemm',
                                 inputs=['1', '2', '3'],
                                 outputs=['y'],
                                 alpha=0.5,
                                 beta=0.8,
                                 transA=1)

    return ([node], [m1, m2, m3], [y])


@onnx_test
def globalavgpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test
def globalmaxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 1, 1])

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test
def group_conv_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 4, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 1, 3, 3])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 4, 14, 14])

    node = onnx.helper.make_node(
        'Conv',
        inputs=['0', '1'],
        group=4,
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test
def imagescaler_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 16, 16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 16, 16])

    node = onnx.helper.make_node('ImageScaler',
                                 inputs=['0'],
                                 outputs=['1'],
                                 bias=[0.01, 0.02, 0.03],
                                 scale=0.5)

    return ([node], [x], [y])


@onnx_test
def implicit_add_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4, 1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    return ([node], [x, y], [z])


@onnx_test
def implicit_pow_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4, 1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test
def implicit_sub_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test
def leaky_relu_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node('LeakyRelu',
                                 inputs=['0'],
                                 outputs=['1'],
                                 alpha=0.01)

    return ([node], [x], [y])


@onnx_test
def log_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Log',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def logsoftmax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5, 6])

    node = onnx.helper.make_node('LogSoftmax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axis=1)

    return ([node], [x], [y])


@onnx_test
def lrn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 28, 24, 24])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 28, 24, 24])

    node = onnx.helper.make_node('LRN',
                                 inputs=['0'],
                                 size=5,
                                 alpha=0.0001,
                                 beta=0.75,
                                 bias=1.0,
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test
def matmul_bmbm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 2, 1, 7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 2, 3, 6, 8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test
def matmul_bmv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 6])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test
def matmul_mv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [6])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test
def matmul_vbm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test
def matmul_vm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test
def matmul_vv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    return ([node], [m1, m2], [y])


@onnx_test
def max_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Max',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test
def min_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Min',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test
def no_pad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[0, 0, 0, 0],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test
def pad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node('Pad',
                                 inputs=['0'],
                                 pads=[1, 1, 1, 1],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test
def pow_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test
def reducemax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes = [2]

    node = onnx.helper.make_node('ReduceMax',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def reducemean_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    axes = [2, 3]

    node = onnx.helper.make_node('ReduceMean',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def reducemean_keepdims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes = [2]

    node = onnx.helper.make_node('ReduceMean',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test
def reducemin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 1, 5, 1])
    axes = [1, 3]

    node = onnx.helper.make_node('ReduceMin',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test
def reducesum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes = [2]

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test
def reducesum_multiaxis_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes = [2, 3]

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=0)

    return ([node], [x], [y])


@onnx_test
def reducesum_keepdims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes = [2, 3]

    node = onnx.helper.make_node('ReduceSum',
                                 inputs=['x'],
                                 outputs=['y'],
                                 axes=axes,
                                 keepdims=1)

    return ([node], [x], [y])


@onnx_test
def reshape_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 2, 3])
    x_shape = helper.make_tensor_value_info('1', TensorProto.INT64, [2])
    x_shape_list = [3, 8]
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 8])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3, 8])

    node = onnx.helper.make_node('Reshape', inputs=['0', '1'], outputs=['2'])

    node2 = onnx.helper.make_node('Reshape',
                                  inputs=['0'],
                                  shape=x_shape_list,
                                  outputs=['3'])

    return ([node, node2], [x, x_shape], [y, y2],
            [helper.make_tensor('1', TensorProto.INT64, [2], [3, 8])])


@onnx_test
def reshape_non_standard_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    trans_x = helper.make_tensor_value_info('trans_x', TensorProto.FLOAT,
                                            [2, 4, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 3, 2])

    trans = helper.make_node(
        'Transpose',
        inputs=['x'],
        outputs=['trans_x'],
        perm=[0, 2, 1],
    )

    res = onnx.helper.make_node('Reshape',
                                inputs=['trans_x'],
                                outputs=['y'],
                                shape=[4, 3, 2])

    return ([trans, res], [x], [y])


@onnx_test
def shape_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def shape_gather_test():
    values = np.array([1])
    value = helper.make_tensor_value_info('value', TensorProto.INT32, [1])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [7, 3, 10])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1])

    value_tensor = helper.make_tensor(name='const_tensor',
                                      data_type=TensorProto.INT32,
                                      dims=values.shape,
                                      vals=values.flatten().astype(int))

    node_const = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['value'],
        value=value_tensor,
    )

    node_shape = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
    )

    node_gather = helper.make_node(
        'Gather',
        inputs=['y', 'value'],
        outputs=['z'],
        axis=0,
    )

    return ([node_const, node_shape, node_gather], [x], [z])


@onnx_test
def sign_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [10, 5])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [10, 5])

    node = onnx.helper.make_node(
        'Sign',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def sin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Sin',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def sinh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Sinh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def slice_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node('Slice',
                                 inputs=['0'],
                                 axes=[0, 1],
                                 starts=[1, 0],
                                 ends=[2, 2],
                                 outputs=['1'])

    return ([node], [x], [y])


@onnx_test
def softmax_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3])

    node = onnx.helper.make_node('Softmax', inputs=['0'], outputs=['1'])

    return ([node], [x], [y])


@onnx_test
def sqrt_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 15])

    node = onnx.helper.make_node(
        'Sqrt',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def squeeze_unsqueeze_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                      [1, 3, 1, 1, 2, 1])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT,
                                      [1, 1, 3, 1, 2, 1])

    node = onnx.helper.make_node('Squeeze',
                                 inputs=['0'],
                                 axes=[0, 2, 3, 5],
                                 outputs=['1'])

    node2 = onnx.helper.make_node('Unsqueeze',
                                  inputs=['1'],
                                  axes=[0, 1, 3, 5],
                                  outputs=['2'])

    return ([node, node2], [x], [z])


@onnx_test
def sub_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', '1'],
        outputs=['out'],
        broadcast=1,
        axis=1,
    )

    return ([node], [arg0, arg1], [arg_out])


@onnx_test
def sub_scalar_test():
    values = np.array([1])
    arg_node = helper.make_tensor_value_info('0', TensorProto.FLOAT,
                                             [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                            [2, 3, 4, 5])

    values_tensor = helper.make_tensor(name='const',
                                       data_type=TensorProto.FLOAT,
                                       dims=values.shape,
                                       vals=values.flatten().astype(float))

    arg_const = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['arg_const'],
        value=values_tensor,
    )

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', 'arg_const'],
        outputs=['out'],
    )

    return ([arg_const, node], [arg_node], [arg_out])


@onnx_test
def sum_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])

    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Sum',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test
def sum_test():
    a = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Sum',
        inputs=['0', '1', '2'],
        outputs=['3'],
    )

    return ([node], [a, b, c], [y])


@onnx_test
def tan_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Tan',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def tanh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Tanh',
        inputs=['x'],
        outputs=['y'],
    )

    return ([node], [x], [y])


@onnx_test
def transpose_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

    node = onnx.helper.make_node(
        'Transpose',
        perm=[0, 3, 1, 2],
        inputs=['0'],
        outputs=['1'],
    )

    return ([node], [x], [y])


@onnx_test
def transpose_gather_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 5, 4, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32,
                                      [2, 4, 3, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT,
                                      [3, 2, 3, 4, 5, 4, 5, 6])

    td = onnx.helper.make_node(
        'Transpose',
        inputs=['data'],
        outputs=['tdata'],
        perm=[0, 2, 1, 3],
    )

    ti = onnx.helper.make_node('Transpose',
                               inputs=['indices'],
                               outputs=['tindices'],
                               perm=[0, 2, 1, 3])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['tdata', 'tindices'],
        outputs=['y'],
        axis=1,
    )

    return ([td, ti, node], [x, i], [y])


@onnx_test
def unknown_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])
    a = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node('Unknown', inputs=['0', '1'], outputs=['2'])

    node2 = onnx.helper.make_node('Unknown', inputs=['2'], outputs=['3'])

    return ([node, node2], [x, y], [a])

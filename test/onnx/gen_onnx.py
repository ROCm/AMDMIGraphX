import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

def acos_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Acos',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_acos',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='acos-example')
    onnx.save(model_def, 'onnx_acos.onnx')

def add_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4,5])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        broadcast=1,
        axis=1,
        outputs=['2']
    )

    graph_def = helper.make_graph(
        [node],
        'test-add_bcast',
        [x,y],
        [z]
    )

    model_def = helper.make_model(graph_def, producer_name='add_bcast-example')
    onnx.save(model_def, 'add_bcast_test.onnx')

def add_fp16_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT16, [1])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT16, [1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT16, [1])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    graph_def = helper.make_graph(
        [node],
        'test-add-fp16',
        [x,y],
        [z],
        # '0' -> 1.5, '1' -> 2.5
        initializer=[onnx.helper.make_tensor('0', TensorProto.FLOAT16, [1], [15872]),
                    onnx.helper.make_tensor('1', TensorProto.FLOAT16, [1], [16640])]
    )

    model_def = helper.make_model(graph_def, producer_name=('add-fp16-example'))
    onnx.save(model_def, 'add_fp16_test.onnx')

def add_scalar_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4,5])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2']
    )

    graph_def = helper.make_graph(
        [node],
        'test-add-scalar',
        [x,y],
        [z],
        initializer=[helper.make_tensor('1', TensorProto.FLOAT, [], [1])]
    )

    model_def = helper.make_model(graph_def, producer_name='add_scalar-example')
    onnx.save(model_def, 'add_scalar_test.onnx')

def argmax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 6])

    node = onnx.helper.make_node(
        'ArgMax',
        inputs=['x'],
        outputs=['y'],
        axis=2,
        keepdims = 0
    )


    graph_def = helper.make_graph(
        [node],
        'test_argmax',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='argmax-example')
    onnx.save(model_def, 'argmax_test.onnx')

def argmin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])

    node = onnx.helper.make_node(
        'ArgMin',
        inputs=['x'],
        outputs=['y'],
        axis=3,
        keepdims = 0
    )

    graph_def = helper.make_graph(
        [node],
        'test_argmin',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='argmin-example')
    onnx.save(model_def, 'argmin_test.onnx')

def asin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Asin',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_asin',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='asin-example')
    onnx.save(model_def, 'asin_test.onnx')

def atan_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Atan',
        inputs=['x'],
        outputs=['y'],
    )
    
    graph_def = helper.make_graph(
        [node],
        'test_atan',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='atan-example')
    onnx.save(model_def, 'atan_test.onnx')

def cast_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Cast',
        inputs=['x'],
        outputs=['y'],
        to = 1
    )
    
    graph_def = helper.make_graph(
        [node],
        'test_cast',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='cast-example')
    onnx.save(model_def, 'cast_test.onnx')

def clip_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])


    node = onnx.helper.make_node(
        'Clip',
        inputs=['0'],
        outputs=['1'],
        max=6.0,
        min=0.0
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='clip-example')
    onnx.save(model_def, 'clip_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-concat',
        [x,y],
        [z]
    )

    model_def = helper.make_model(graph_def, producer_name='concat-example')
    onnx.save(model_def, 'concat_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-constant',
        [],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name=('constant-example'))
    onnx.save(model_def, 'constant_test.onnx')

def constant_fill_test():
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'ConstantFill',
        inputs=[],
        outputs=['value'],
        dtype = 1,
        value = 1.0,
        shape = [2, 3],
        input_as_shape = 0,
    )

    graph_def = helper.make_graph(
        [node],
        'constant_fill',
        [],
        [value],
    )

    model_def = helper.make_model(graph_def, producer_name='constant-fill-example')
    onnx.save(model_def, 'constant_fill_test.onnx')

def constant_fill_input_as_shape_test():
    np_shape = np.array([2, 3])
    shape = helper.make_tensor_value_info('shape', TensorProto.INT32, [2])
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [2, 3])

    ts_shape = helper.make_tensor(
        name = 'shape_tensor',
        data_type = TensorProto.INT32,
        dims = np_shape.shape,
        vals = np_shape.flatten().astype(int)
    )

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
        dtype = 1,
        value = 1.0,
        input_as_shape = 1,
    )

    graph_def = helper.make_graph(
        [const_shape_node, node],
        'constant_fill',
        [],
        [value],
    )

    model_def = helper.make_model(graph_def, producer_name='constant-fill-example')
    onnx.save(model_def, 'constant_fill_input_as_shape_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-constant',
        [],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name=('constant-scalar-example'))
    onnx.save(model_def, 'constant_scalar_test.onnx')

def const_of_shape_empty_input_test():
    tensor_val = onnx.helper.make_tensor(
        'value',
        onnx.TensorProto.INT64, [1],[10]
    )
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    empty_val = np.array([]).astype(np.int64)
    empty_ts = helper.make_tensor(
        name='empty_tensor',
        data_type = TensorProto.INT32,
        dims=empty_val.shape,
        vals=empty_val.flatten().astype(int)
    )
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
        value = tensor_val,
    )

    graph_def = helper.make_graph(
        [shape_const, node],
        'constant_of_shape',
        [],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='constant-of-shape')
    onnx.save(model_def, 'const_of_shape_empty_input_test.onnx')

def const_of_shape_float_test():
    tensor_val = onnx.helper.make_tensor(
        'value',
        onnx.TensorProto.FLOAT, [1],[10])

    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(
        name = 'shape_tensor',
        data_type = TensorProto.INT32,
        dims = shape_val.shape,
        vals = shape_val.flatten().astype(int)
    )

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
        value = tensor_val
    )

    graph_def = helper.make_graph(
            [shape_const, node],
            'constant_of_shape',
            [],
            [y],
    )

    model_def = helper.make_model(graph_def, producer_name='constant-of-shape')
    onnx.save(model_def, 'const_of_shape_float_test.onnx')

def const_of_shape_int64_test():
    tensor_val = onnx.helper.make_tensor(
        'value',
        onnx.TensorProto.INT64, [1],[10]
    )
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(
        name = 'shape_tensor',
        data_type = TensorProto.INT32,
        dims = shape_val.shape,
        vals = shape_val.flatten().astype(int)
    )
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
        value = tensor_val
    )

    graph_def = helper.make_graph(
        [shape_const, node],
        'constant_of_shape',
        [],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='constant-of-shape')
    onnx.save(model_def, 'const_of_shape_int64_test.onnx')

def const_of_shape_no_value_attr_test():
    tensor_val = onnx.helper.make_tensor(
        'value',
        onnx.TensorProto.INT64, [1],[10]
    )
    shape_val = np.array([2, 3, 4]).astype(np.int64)
    shape_ts = helper.make_tensor(
        name = 'shape_tensor',
        data_type = TensorProto.INT32,
        dims = shape_val.shape,
        vals = shape_val.flatten().astype(int)
    )
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

    graph_def = helper.make_graph(
        [shape_const, node],
        'constant_of_shape',
        [],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='constant-of-shape')
    onnx.save(model_def, 'const_of_shape_no_value_attr_test.onnx')

def cos_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Cos',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_cos',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='cos-example')
    onnx.save(model_def, 'cos_test.onnx')

def cosh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'Cosh',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_cosh',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='cosh-example')
    onnx.save(model_def, 'cosh_test.onnx')    

def dropout_test():
        x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 2, 2])
        y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

        node = onnx.helper.make_node(
            'Dropout',
            inputs=['0'],
            outputs=['1'],
        )

        graph_def = helper.make_graph(
            [node],
            'test-dropout',
            [x],
            [y]
        )

        model_def = helper.make_model(graph_def, producer_name='dropout-example')
        onnx.save(model_def, 'dropout_test.onnx')

def elu_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'Elu',
        inputs=['0'],
        outputs=['1'],
        alpha=0.01
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='elu-example')
    onnx.save(model_def, 'elu_test.onnx')

def erf_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 15])

    node = onnx.helper.make_node(
        'Erf',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_erf',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='erf-example')
    onnx.save(model_def, 'erf_test.onnx')

def exp_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Exp',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_exp',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='exp-example')
    onnx.save(model_def, 'exp_test.onnx')

def expand_test():
    shape_val = np.array([2, 3, 4, 5]).astype(np.int64)
    shape_ts = helper.make_tensor(
        name = 'shape_tensor',
        data_type = TensorProto.INT32,
        dims = shape_val.shape,
        vals = shape_val.flatten().astype(int)
    )
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape'],
        value=shape_ts,
    )
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])
    
    node = onnx.helper.make_node(
        'Expand',
        inputs=['x', 'shape'],
        outputs=['y']
    )

    graph_def = helper.make_graph(
        [shape_const, node],
        'expand',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='expand')
    onnx.save(model_def, 'expand_test.onnx')

def flatten_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [6,  20])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 60])

    node = onnx.helper.make_node(
        'Flatten',
        inputs=['0'],
        axis=2,
        outputs=['2']
    )

    node2 = onnx.helper.make_node(
        'Flatten',
        inputs=['0'],
        outputs=['3']
    )

    graph_def = helper.make_graph(
        [node,node2],
        'test-flatten',
        [x],
        [y,y2]
    )

    model_def = helper.make_model(graph_def, producer_name=('flatten-example'))
    onnx.save(model_def, 'flatten_test.onnx')

def gather_test():
    x = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5, 6])
    i = helper.make_tensor_value_info('indices', TensorProto.INT32, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Gather',
        inputs=['data', 'indices'],
        outputs=['y'],
        axis=1,
    )

    graph_def = helper.make_graph(
        [node],
        'test_gather',
        [x, i],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='gather-example')
    onnx.save(model_def, 'gather_test.onnx')

def gemm_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [5, 7])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [11, 5])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [])
    a = helper.make_tensor_value_info('3', TensorProto.FLOAT, [7, 11])

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['0', '1', '2'],
        outputs=['3'],
        alpha=2.0,
        beta=2.0,
        transA=1,
        transB=1
    )

    graph_def = helper.make_graph(
        [node],
        'test-gemm',
        [x, y, z],
        [a]
    )

    model_def = helper.make_model(graph_def, producer_name=('gemm-example'))
    onnx.save(model_def, 'gemm_test.onnx')

def gemm_ex_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 1, 5, 6])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 5, 7])
    m3 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 1, 6, 7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 6, 7])

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['1', '2', '3'],
        outputs=['y'],
        alpha = 0.5,
        beta = 0.8,
        transA = 1
    )

    graph_def = helper.make_graph(
        [node],
        'test_gemm_ex',
        [m1, m2, m3],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='gemm-example')
    onnx.save(model_def, 'gemm_ex_test.onnx')

def gemm_ex_brcst_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 1, 5, 6])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 5, 7])
    m3 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [1, 1, 6, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 6, 7])

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['1', '2', '3'],
        outputs=['y'],
        alpha = 0.5,
        beta = 0.8,
        transA = 1
    )

    graph_def = helper.make_graph(
        [node],
        'test_gemm_ex',
        [m1, m2, m3],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='gemm-example')
    onnx.save(model_def, 'gemm_ex_brcst_test.onnx')

def globalavgpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1,3,16,16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1,3,1,1])

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=['0'],
        outputs=['1'],
    )

    graph_def = helper.make_graph(
        [node],
        'test-globalavgpool',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='globalavgpool-example')
    onnx.save(model_def, 'globalavgpool_test.onnx')

def globalmaxpool_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1,3,16,16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1,3,1,1])

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['0'],
        outputs=['1'],
    )

    graph_def = helper.make_graph(
        [node],
        'test-globalmaxpool',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='globalmaxpool-example')
    onnx.save(model_def, 'globalmaxpool_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-group_conv',
        [x,y],
        [z]
    )

    model_def = helper.make_model(graph_def, producer_name='group_conv-example')
    onnx.save(model_def, 'group_conv_test.onnx')

def imagescaler_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1,3,16,16])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1,3,16,16])

    node = onnx.helper.make_node(
        'ImageScaler',
        inputs=['0'],
        outputs=['1'],
        bias=[0.01,0.02,0.03],
        scale=0.5
    )

    graph_def = helper.make_graph(
        [node],
        'test-imagescaler',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='imagescaler-example')
    onnx.save(model_def, 'imagescaler_test.onnx')

def implicit_add_bcast_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4, 1])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Add',
        inputs=['0', '1'],
        outputs=['2'],
    )

    graph_def = helper.make_graph(
        [node],
        'test-multi_bcast',
        [x,y],
        [z]
    )

    model_def = helper.make_model(graph_def, producer_name='implicit_bcast-example')
    onnx.save(model_def, 'implicit_add_bcast_test.onnx')

def implicit_pow_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4, 1])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )

    graph_def = helper.make_graph(
        [node],
        'pow_test',
        [arg0, arg1],
        [arg_out],
    )

    model_def = helper.make_model(graph_def, producer_name='pow2')
    onnx.save(model_def, 'implicit_pow_bcast_test.onnx')

def implicit_sub_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', '1'],
        outputs=['out'],
    )

    graph_def = helper.make_graph(
        [node],
        'subtraction2',
        [arg0, arg1],
        [arg_out],
    )

    model_def = helper.make_model(graph_def, producer_name='add2')
    onnx.save(model_def, 'implicit_sub_bcast_test.onnx')

def leaky_relu_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3])

    node = onnx.helper.make_node(
        'LeakyRelu',
        inputs=['0'],
        outputs=['1'],
        alpha=0.01
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='leaky_relu-example')
    onnx.save(model_def, 'leaky_relu_test.onnx')

def log_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Log',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_log',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='log-example')
    onnx.save(model_def, 'log_test.onnx')

def logsoftmax_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5, 6])

    node = onnx.helper.make_node(
        'LogSoftmax',
        inputs=['x'],
        outputs=['y'],
        axis = 1
    )

    graph_def = helper.make_graph(
        [node],
        'test_logsoftmax',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='logsoftmax-example')
    onnx.save(model_def, 'logsoftmax_test.onnx')

def lrn_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 28, 24, 24])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 28, 24, 24])

    node = onnx.helper.make_node(
        'LRN',
        inputs=['0'],
        size=5,
        alpha=0.0001,
        beta=0.75,
        bias=1.0,
        outputs=['1']
    )

    graph_def = helper.make_graph(
        [node],
        'test-lrn',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name=('lrn-example'))
    onnx.save(model_def, 'lrn_test.onnx')

def matmul_bmbm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 2, 1, 7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 2, 3, 6, 8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_matmul',
        [m1, m2],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='matmul-example')
    onnx.save(model_def, 'matmul_bmbm_test.onnx')

def matmul_bmv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 6])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_matmul',
        [m1, m2],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='matmul-example')
    onnx.save(model_def, 'matmul_bmv_test.onnx')

def matmul_mv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [6, 7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [6])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_matmul',
        [m1, m2],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='matmul-example')
    onnx.save(model_def, 'matmul_mv_test.onnx')

def matmul_vbm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [5, 7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_matmul',
        [m1, m2],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='matmul-example')
    onnx.save(model_def, 'matmul_vbm_test.onnx')

def matmul_vm_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [8])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_matmul',
        [m1, m2],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='matmul-example')
    onnx.save(model_def, 'matmul_vm_test.onnx')

def matmul_vv_test():
    m1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [7])
    m2 = helper.make_tensor_value_info('2', TensorProto.FLOAT, [7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['1', '2'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_matmul',
        [m1, m2],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='matmul-example')
    onnx.save(model_def, 'matmul_vv_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-dropout',
        [a, b, c],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='max-example')
    onnx.save(model_def, 'max_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-dropout',
        [a, b, c],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='min-example')
    onnx.save(model_def, 'min_test.onnx')

def no_pad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 2])

    node = onnx.helper.make_node(
        'Pad',
        inputs=['0'],
        pads=[0,0,0,0],
        outputs=['1']
    )


    graph_def = helper.make_graph(
        [node],
        'test-no-pad',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='no-pad-example')
    onnx.save(model_def, 'no_pad_test.onnx')

def pad_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [4, 4])

    node = onnx.helper.make_node(
        'Pad',
        inputs=['0'],
        pads=[1,1,1,1],
        outputs=['1']
    )


    graph_def = helper.make_graph(
        [node],
        'test-pad',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='pad-example')
    onnx.save(model_def, 'pad_test.onnx')

def pow_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Pow',
        inputs=['0', '1'],
        outputs=['out'],
    )


    graph_def = helper.make_graph(
        [node],
        'pow_test',
        [arg0, arg1],
        [arg_out],
    )

    model_def = helper.make_model(graph_def, producer_name='pow2')
    onnx.save(model_def, 'pow_test.onnx')

def reducemean_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    axes=[2, 3]

    node = onnx.helper.make_node(
        'ReduceMean',
        inputs=['x'],
        outputs=['y'],
        axes=axes,
        keepdims = 0
    )

    graph_def = helper.make_graph(
        [node],
        'test_reducemean',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='reducemean-example')
    onnx.save(model_def, 'reducemean_test.onnx')

def reducemean_keepdims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 6])
    axes=[2]

    node = onnx.helper.make_node(
        'ReduceMean',
        inputs=['x'],
        outputs=['y'],
        axes=axes,
        keepdims = 1
    )

    graph_def = helper.make_graph(
        [node],
        'test_reducemean',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='reducemean-example')
    onnx.save(model_def, 'reducemean_keepdims_test.onnx')

def reducesum_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes=[2]

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['x'],
        outputs=['y'],
        axes=axes,
        keepdims = 0
    )

    graph_def = helper.make_graph(
        [node],
        'test_reducesum',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='reducesum-example')
    onnx.save(model_def, 'reducesum_test.onnx')

def reducesum_multiaxis_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes=[2, 3]

    node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['x'],
            outputs=['y'],
            axes=axes,
            keepdims = 1
            )

    graph_def = helper.make_graph(
            [node],
            'test_reducesum',
            [x],
            [y],
    )

    model_def = helper.make_model(graph_def, producer_name='reducesum-example')
    onnx.save(model_def, 'reducesum_multiaxis_test.onnx')

def reducesum_keepdims_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 1, 1])
    axes=[2, 3]

    node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['x'],
            outputs=['y'],
            axes=axes,
            keepdims = 1
            )

    graph_def = helper.make_graph(
            [node],
            'test_reducesum',
            [x],
            [y],
    )

    model_def = helper.make_model(graph_def, producer_name='reducesum-example')
    onnx.save(model_def, 'reducesum_keepdims_test.onnx')

def reshape_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [4, 2, 3])
    x_shape = helper.make_tensor_value_info('1', TensorProto.INT64, [2])
    x_shape_list = [3,8]
    y = helper.make_tensor_value_info('2', TensorProto.FLOAT, [3, 8])
    y2 = helper.make_tensor_value_info('3', TensorProto.FLOAT, [3, 8])

    node = onnx.helper.make_node(
        'Reshape',
        inputs=['0', '1'],
        outputs=['2']
    )

    node2 = onnx.helper.make_node(
        'Reshape',
        inputs=['0'],
        shape=x_shape_list,
        outputs=['3']
    )

    graph_def = helper.make_graph(
        [node,node2],
        'test-reshape',
        [x, x_shape],
        [y,y2],
        initializer=[helper.make_tensor('1', TensorProto.INT64, [2], [3, 8])]
    )

    model_def = helper.make_model(graph_def, producer_name=('reshape-example'))
    onnx.save(model_def, 'reshape_test.onnx')

def reshape_non_standard_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    trans_x = helper.make_tensor_value_info('trans_x', TensorProto.FLOAT, [2, 4, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 3, 2])

    trans = helper.make_node(
        'Transpose',
        inputs=['x'],
        outputs=['trans_x'],
        perm=[0, 2, 1],
    )

    res = onnx.helper.make_node(
        'Reshape',
        inputs=['trans_x'],
        outputs=['y'],
        shape=[4, 3, 2]
    )

    graph_def = helper.make_graph(
        [trans, res],
        'reshape-ns',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='reshape')
    onnx.save(model_def, 'reshape_non_standard_test.onnx')

def shape_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5, 6])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [4])

    node = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_shape',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='shape-example')
    onnx.save(model_def, 'shape_test.onnx')

def shape_gather_test():
    values = np.array([1])
    value = helper.make_tensor_value_info('value', TensorProto.INT32, [1])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [7, 3, 10])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [3])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1])

    value_tensor = helper.make_tensor(
            name = 'const_tensor',
            data_type = TensorProto.INT32,
            dims = values.shape,
            vals = values.flatten().astype(int))

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

    graph_def = helper.make_graph(
        [node_const, node_shape, node_gather],
        'shape_gather',
        [x],
        [z],
    )

    model_def = helper.make_model(graph_def, producer_name='shape-gather-example')
    onnx.save(model_def, 'shape_gather_test.onnx')

def sign_test():
    x = helper.make_tensor_value_info('x', TensorProto.DOUBLE, [10, 5])
    y = helper.make_tensor_value_info('y', TensorProto.DOUBLE, [10, 5])

    node = onnx.helper.make_node(
        'Sign',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_sign',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='sign-example')
    onnx.save(model_def, 'sign_test.onnx')

def sin_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Sin',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_sin',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='sin-example')
    onnx.save(model_def, 'sin_test.onnx')

def sinh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
        'Sinh',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_sinh',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='sinh-example')
    onnx.save(model_def, 'sinh_test.onnx')

def slice_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [3, 2])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node(
        'Slice',
        inputs=['0'],
        axes=[0, 1],
        starts=[1,0],
        ends=[2, 2],
        outputs=['1']
    )

    graph_def = helper.make_graph(
        [node],
        'test-slice',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name=('slice-example'))
    onnx.save(model_def, 'slice_test.onnx')

def softmax_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3])

    node = onnx.helper.make_node(
        'Softmax',
        inputs=['0'],
        outputs=['1']
    )

    graph_def = helper.make_graph(
        [node],
        'test-softmax',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name=('softmax-example'))
    onnx.save(model_def, 'softmax_test.onnx')

def sqrt_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 15])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 15])

    node = onnx.helper.make_node(
        'Sqrt',
        inputs=['x'],
        outputs=['y'],
    )

    graph_def = helper.make_graph(
        [node],
        'test_sqrt',
        [x],
        [y],
    )

    model_def = helper.make_model(graph_def, producer_name='sqrt-example')
    onnx.save(model_def, 'sqrt_test.onnx')

def squeeze_unsqueeze_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 3, 1, 1, 2, 1])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [1, 1, 3, 1, 2, 1])

    node = onnx.helper.make_node(
        'Squeeze',
        inputs=['0'],
        axes=[0, 2, 3, 5],
        outputs=['1']
    )

    node2 = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['1'],
        axes=[0, 1, 3, 5],
        outputs=['2']
    )

    graph_def = helper.make_graph(
        [node,node2],
        'test-squeeze-unsqueeze',
        [x],
        [z]
    )

    model_def = helper.make_model(graph_def, producer_name=('squeeze-unsqueeze-example'))
    onnx.save(model_def, 'squeeze_unsqueeze_test.onnx')

def sub_bcast_test():
    arg0 = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg1 = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Sub',
        inputs=['0', '1'],
        outputs=['out'],
        broadcast = 1,
        axis = 1,
    )


    graph_def = helper.make_graph(
        [node],
        'subtraction2',
        [arg0, arg1],
        [arg_out],
    )

    model_def = helper.make_model(graph_def, producer_name='subtraction2')
    onnx.save(model_def, 'sub_bcast_test.onnx')

def sub_scalar_test():
    values = np.array([1])
    arg_node = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    arg_out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 3, 4, 5])

    values_tensor = helper.make_tensor(
        name = 'const',
        data_type = TensorProto.FLOAT,
        dims = values.shape,
        vals = values.flatten().astype(float)
    )

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

    graph_def = helper.make_graph(
        [arg_const, node],
        'subtraction1',
        [arg_node],
        [arg_out],
    )

    model_def = helper.make_model(graph_def, producer_name='subtraction1')
    onnx.save(model_def, 'sub_scalar_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-sum',
        [a, b, c],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='sum-example')
    onnx.save(model_def, 'sum_test.onnx')

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

    graph_def = helper.make_graph(
        [node],
        'test-sum',
        [a, b, c],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='sum-example')
    onnx.save(model_def, 'sum_test.onnx')

def tan_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10])

    node = onnx.helper.make_node(
            'Tan',
            inputs=['x'],
            outputs=['y'],
            )

    graph_def = helper.make_graph(
            [node],
            'test_tan',
            [x],
            [y],
    )

    model_def = helper.make_model(graph_def, producer_name='tan-example')
    onnx.save(model_def, 'tan_test.onnx')

def tanh_test():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])

    node = onnx.helper.make_node(
            'Tanh',
            inputs=['x'],
            outputs=['y'],
            )

    graph_def = helper.make_graph(
            [node],
            'test_tanh',
            [x],
            [y],
    )

    model_def = helper.make_model(graph_def, producer_name='tanh-example')
    onnx.save(model_def, 'tahn_test.onnx')

def transpose_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [1, 2, 2, 3])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [1, 3, 2, 2])

    node = onnx.helper.make_node(
        'Transpose',
        perm=[0, 3, 1, 2],
        inputs=['0'],
        outputs=['1'],
    )

    graph_def = helper.make_graph(
        [node],
        'test-transpose',
        [x],
        [y]
    )

    model_def = helper.make_model(graph_def, producer_name='transpose-example')
    onnx.save(model_def, 'transpose_test.onnx')

def unknown_test():
    x = helper.make_tensor_value_info('0', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT, [3, 4])
    z = helper.make_tensor_value_info('2', TensorProto.FLOAT, [2, 3, 4, 5])
    a = helper.make_tensor_value_info('3', TensorProto.FLOAT, [2, 3, 4, 5])

    node = onnx.helper.make_node(
        'Unknown',
        inputs=['0', '1'],
        outputs=['2']
    )

    node2 = onnx.helper.make_node(
        'Unknown',
        inputs=['2'],
        outputs=['3']
    )

    graph_def = helper.make_graph(
        [node,node2],
        'test-unknown',
        [x,y],
        [a]
    )

    model_def = helper.make_model(graph_def, producer_name='unknown-example')
    onnx.save(model_def, 'unknown_test.onnx')
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2


def tf_test(op_test):
    def run_test():
        g1 = tf.Graph()
        op_test(g1)
        tf.io.write_graph(g1,
                          '.',
                          '{}.pb'.format(op_test.__name__),
                          as_text=False)

    return run_test


@tf_test
def add_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='1')
        tf.add(g1_input, g2_input, name='add1')


@tf_test
def add_bcast_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(2, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(2, 1), name='1')
        tf.math.add(g1_input, g2_input, name='add_bcast1')


@tf_test
def argmax_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(3, 4, 5, 6), name='0')
        tf.argmax(g1_input, axis=2, name='argmax1')


@tf_test
def argmin_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(3, 4, 5, 6), name='0')
        tf.argmin(g1_input, axis=2, name='argmin1')


@tf_test
def assert_less_equal_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(2, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(2, 3), name='1')
        with tf.control_dependencies(
            [tf.assert_less_equal(g1_input, g2_input)]):
            tf.add(g1_input, g2_input, name='add1')


@tf_test
def batchmatmul_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 2, 8, 4), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 2, 4, 8), name='1')
        tf.matmul(g1_input,
                  g2_input,
                  transpose_a=True,
                  transpose_b=True,
                  name='batchmatmul1')


@tf_test
def batchnorm_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 16, 16, 32), name='0')
        g1_scale = tf.constant(1.0, dtype=tf.float32, shape=[32], name='1')
        g1_offset = tf.placeholder(tf.float32, shape=(32), name='2')
        g1_mean = tf.placeholder(tf.float32, shape=(32), name='3')
        g1_variance = tf.placeholder(tf.float32, shape=(32), name='4')
        tf.nn.fused_batch_norm(g1_input,
                               g1_scale,
                               g1_offset,
                               g1_mean,
                               g1_variance,
                               epsilon=0.00001,
                               is_training=False,
                               name='batchnorm1')


@tf_test
def biasadd_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 500), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(500), name='1')
        tf.nn.bias_add(g1_input, g2_input, name='bias_add1')


@tf_test
def cast_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.cast(g1_input, dtype=tf.int32, name='cast1')


@tf_test
def concat_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(4, 7, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(4, 2, 3), name='1')
        tf.concat([g1_input, g2_input], axis=1, name='concat1')


@tf_test
def const_test(g1):
    with g1.as_default():
        tf.constant(1.0, dtype=tf.float32, name='constant1')


@tf_test
def conv_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 16, 16, 3), name='0')
        g1_weights = tf.constant(value=1.0,
                                 dtype=tf.float32,
                                 shape=(3, 3, 3, 32),
                                 name='1')
        tf.nn.conv2d(g1_input, g1_weights, [1, 1, 1, 1], "SAME", name='conv1')


@tf_test
def depthwiseconv_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 16, 16, 3), name='0')
        g1_weights = tf.constant(value=1.0,
                                 dtype=tf.float32,
                                 shape=(3, 3, 3, 1),
                                 name='1')
        tf.nn.depthwise_conv2d_native(g1_input,
                                      g1_weights, [1, 1, 1, 1],
                                      "SAME",
                                      name='depthwiseconv1')


@tf_test
def expanddims_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(2, 3, 4), name='0')
        tf.expand_dims(g1_input, axis=-1, name='expanddims_neg')


@tf_test
def gather_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(2, 4), name='0')
        tf.gather(g1_input, [1, 1], axis=1, name='gather1')


@tf_test
def identity_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.identity(g1_input, 'identity')


@tf_test
def matmul_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(8, 4), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(4, 8), name='1')
        tf.matmul(g1_input,
                  g2_input,
                  transpose_a=True,
                  transpose_b=True,
                  name='matmul1')


@tf_test
def mean_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.math.reduce_mean(g1_input, axis=(2, 3), keepdims=True, name='mean1')
        tf.math.reduce_mean(g1_input,
                            axis=(2, 3),
                            keepdims=False,
                            name='mean2')


@tf_test
def mean_test_nhwc(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 16, 16, 3), name='0')
        tf.math.reduce_mean(g1_input, axis=(1, 2), keepdims=True, name='mean1')
        tf.math.reduce_mean(g1_input,
                            axis=(1, 2),
                            keepdims=False,
                            name='mean2')


@tf_test
def mul_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 16), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 16), name='1')
        tf.multiply(g1_input, g2_input, name='mul1')


@tf_test
def onehot_test(g1):
    with g1.as_default():
        g1_input = tf.constant((1, 1, 1, 1, 1), dtype=tf.int32)
        tf.one_hot(g1_input, 2, name='onehot1')


@tf_test
def pack_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(2), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(2), name='1')
        g3_input = tf.placeholder(tf.float32, shape=(2), name='2')
        tf.stack([g1_input, g2_input, g3_input], axis=1, name='pack1')


@tf_test
def pack_test_nhwc(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 2), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 2), name='1')
        g3_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 2), name='2')
        tf.stack([g1_input, g2_input, g3_input], axis=3, name='pack1')


@tf_test
def pooling_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 16, 16, 3), name='0')
        tf.nn.avg_pool(value=g1_input,
                       ksize=(1, 2, 2, 1),
                       strides=(1, 2, 2, 1),
                       padding='VALID',
                       data_format='NHWC',
                       name='avg_pooling')
        tf.nn.max_pool(value=g1_input,
                       ksize=(1, 2, 2, 1),
                       strides=(1, 2, 2, 1),
                       padding='VALID',
                       data_format='NHWC',
                       name='max_pooling')


@tf_test
def pow_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='1')
        tf.pow(g1_input, g2_input, name='pow1')


@tf_test
def relu_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.nn.relu(g1_input, 'relu')


@tf_test
def relu6_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.nn.relu6(g1_input, 'relu6')


@tf_test
def reshape_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(16), name='0')
        tf.reshape(g1_input, (1, 1, 1, 16), 'reshape')


@tf_test
def rsqrt_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.math.rsqrt(g1_input, 'rsqrt')


@tf_test
def shape_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
    g1.create_op(op_type='Shape', inputs=[g1_input])


@tf_test
def slice_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(5, 10), name='0')
        tf.slice(g1_input, [1, 0], [2, -1], name='slice1')


@tf_test
def softmax_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3), name='0')
        tf.nn.softmax(g1_input, name='softmax')


@tf_test
def split_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(5, 30), name='0')
        split0, split1, split2 = tf.split(g1_input, 3, 1, name='split')
        tf.concat([split0, split1], axis=1, name='concat1')
        tf.concat([split1, split2], axis=1, name='concat2')


@tf_test
def split_test_one_output(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(5, 30), name='0')
        tf.split(g1_input, 1, 1, name='split')


@tf_test
def split_test_vector_as_input(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(5, 30), name='0')
        split0, split1, split2 = tf.split(g1_input, [4, 15, 11],
                                          1,
                                          name='split')
        tf.concat([split0, split1], axis=1, name='concat1')
        tf.concat([split1, split2], axis=1, name='concat2')


@tf_test
def sqdiff_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='1')
        tf.squared_difference(g1_input, g2_input, name='sqdiff')


@tf_test
def squeeze_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 2, 3, 1), name='0')
        tf.squeeze(g1_input, name='squeeze')


@tf_test
def stopgradient_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.stop_gradient(g1_input, 'stopgradient')


@tf_test
def stridedslice_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 1, 1, 10), name='0')
        tf.strided_slice(g1_input, [0, 0, 0, 0], [1, 1, 1, 5], [1, 1, 1, 1],
                         shrink_axis_mask=2,
                         name='stridedslice1')


@tf_test
def stridedslice_masks_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 3, 10), name='0')
        tf.strided_slice(g1_input, [0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1],
                         begin_mask=9,
                         end_mask=15,
                         name='stridedslice1')


@tf_test
def sub_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='0')
        g2_input = tf.placeholder(tf.float32, shape=(1, 2, 2, 3), name='1')
        tf.subtract(g1_input, g2_input, name='sub1')


@tf_test
def tanh_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.tanh(g1_input, 'tanh')


@tf_test
def transpose_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(1, 3, 16, 16), name='0')
        tf.transpose(g1_input, perm=[0, 2, 3, 1], name='transpose')


@tf_test
def variable_batch_test(g1):
    with g1.as_default():
        g1_input = tf.placeholder(tf.float32, shape=(0, 3, 16, 16), name='0')
        tf.identity(g1_input, name='identity')

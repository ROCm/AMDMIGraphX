#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import sys
if sys.version_info < (3, 0):
    sys.exit()

import argparse
import os
import unittest
import onnx
import onnx.backend.test
import numpy as np
from onnx_migraphx.backend import MIGraphXBackend as c2
from packaging import version

pytest_plugins = 'onnx.backend.test.report',


class MIGraphXBackendTest(onnx.backend.test.BackendTest):
    def __init__(self, backend, parent_module=None):
        super(MIGraphXBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        prog_string = c2.get_program()
        np.testing.assert_equal(len(ref_outputs),
                                len(outputs),
                                err_msg=prog_string)
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype,
                                    outputs[i].dtype,
                                    err_msg=prog_string)
            if ref_outputs[i].dtype == object:
                np.testing.assert_array_equal(ref_outputs[i],
                                              outputs[i],
                                              err_msg=prog_string)
            else:
                np.testing.assert_allclose(ref_outputs[i],
                                           outputs[i],
                                           rtol=1e-3,
                                           atol=1e-5,
                                           err_msg=prog_string)


def disabled_tests_onnx_1_7_0(backend_test):
    backend_test.exclude(r'test_logsoftmax_axis_0_cpu')
    backend_test.exclude(r'test_logsoftmax_axis_1_cpu')
    backend_test.exclude(r'test_logsoftmax_default_axis_cpu')
    backend_test.exclude(r'test_softmax_axis_0_cpu')
    backend_test.exclude(r'test_softmax_axis_1_cpu')
    backend_test.exclude(r'test_softmax_default_axis_cpu')


def disabled_tests_onnx_1_8_1(backend_test):
    backend_test.exclude(r'test_if_seq_cpu')
    backend_test.exclude(r'test_if_seq_cpu')
    backend_test.exclude(r'test_reduce_sum_default_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_default_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_empty_axes_input_noop_example_cpu')
    backend_test.exclude(r'test_reduce_sum_empty_axes_input_noop_random_cpu')
    backend_test.exclude(r'test_reduce_sum_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_unsqueeze_axis_0_cpu')
    backend_test.exclude(r'test_unsqueeze_axis_1_cpu')
    backend_test.exclude(r'test_unsqueeze_axis_2_cpu')
    backend_test.exclude(r'test_unsqueeze_negative_axes_cpu')
    backend_test.exclude(r'test_unsqueeze_three_axes_cpu')
    backend_test.exclude(r'test_unsqueeze_two_axes_cpu')
    backend_test.exclude(r'test_unsqueeze_unsorted_axes_cpu')


def disabled_tests_onnx_1_10_0(backend_test):
    # unsupported shape attributes
    backend_test.exclude(r'test_shape_end_1_cpu')
    backend_test.exclude(r'test_shape_end_negative_1_cpu')
    backend_test.exclude(r'test_shape_start_1_cpu')
    backend_test.exclude(r'test_shape_start_1_end_2_cpu')
    backend_test.exclude(r'test_shape_start_1_end_negative_1_cpu')
    backend_test.exclude(r'test_shape_start_negative_1_cpu')


def disabled_tests_onnx_1_11_0(backend_test):
    # crash
    backend_test.exclude(r'test_scatter_elements_with_duplicate_indices_cpu')

    # fails
    backend_test.exclude(r'test_roialign_aligned_false_cpu')
    backend_test.exclude(r'test_roialign_aligned_true_cpu')
    backend_test.exclude(r'test_scatternd_add_cpu')
    backend_test.exclude(r'test_scatternd_multiply_cpu')

    # errors
    backend_test.exclude(r'test_identity_opt_cpu')
    backend_test.exclude(r'test_if_opt_cpu')


def disabled_tests_onnx_1_12_0(backend_test):
    pass


def disabled_tests_onnx_1_13_0(backend_test):
    # fails
    backend_test.exclude(r'test_reduce_l1_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_random_cpu')
    backend_test.exclude(r'test_reduce_l1_negative_axes_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_negative_axes_keep_dims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_negative_axes_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_negative_axes_keep_dims_random_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_square_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_square_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_square_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_square_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_scatternd_max_cpu')
    backend_test.exclude(r'test_scatternd_min_cpu')

    # errors
    backend_test.exclude(r'test_constant_pad_axes_cpu')
    backend_test.exclude(r'test_reduce_l1_default_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_l1_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_default_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_l1_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_negative_axes_keep_dims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_negative_axes_keep_dims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_default_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_l2_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_default_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_l2_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_negative_axes_keep_dims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_negative_axes_keep_dims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_asc_axes_cpu')
    backend_test.exclude(r'test_reduce_log_sum_asc_axes_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_default_cpu')
    backend_test.exclude(r'test_reduce_log_sum_default_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_desc_axes_cpu')
    backend_test.exclude(r'test_reduce_log_sum_desc_axes_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_negative_axes_cpu')
    backend_test.exclude(r'test_reduce_log_sum_negative_axes_expanded_cpu')
    backend_test.exclude(r'test_reduce_max_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_max_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_max_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_max_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_max_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_max_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_mean_default_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_default_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_mean_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_mean_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_mean_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_min_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_min_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_min_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_min_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_min_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_min_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_prod_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_prod_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_prod_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_prod_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_prod_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_prod_negative_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_scatter_elements_with_reduction_max_cpu')
    backend_test.exclude(r'test_scatter_elements_with_reduction_min_cpu')

    # The following tests fail due to the CastLike operator being unsupported
    backend_test.exclude(r'test_elu_default_expanded_ver18_cpu')
    backend_test.exclude(r'test_elu_example_expanded_ver18_cpu')
    backend_test.exclude(r'test_elu_expanded_ver18_cpu')
    backend_test.exclude(r'test_hardsigmoid_default_expanded_ver18_cpu')
    backend_test.exclude(r'test_hardsigmoid_example_expanded_ver18_cpu')
    backend_test.exclude(r'test_hardsigmoid_expanded_ver18_cpu')
    backend_test.exclude(r'test_leakyrelu_default_expanded_cpu')
    backend_test.exclude(r'test_leakyrelu_example_expanded_cpu')
    backend_test.exclude(r'test_leakyrelu_expanded_cpu')
    backend_test.exclude(r'test_selu_default_expanded_ver18_cpu')
    backend_test.exclude(r'test_selu_example_expanded_ver18_cpu')
    backend_test.exclude(r'test_selu_expanded_ver18_cpu')
    backend_test.exclude(r'test_thresholdedrelu_default_expanded_ver18_cpu')
    backend_test.exclude(r'test_thresholdedrelu_example_expanded_ver18_cpu')
    backend_test.exclude(r'test_thresholdedrelu_expanded_ver18_cpu')
    backend_test.exclude(r'test_relu_expanded_ver18_cpu')
    backend_test.exclude(r'test_softsign_example_expanded_ver18_cpu')
    backend_test.exclude(r'test_softsign_expanded_ver18_cpu')


def disabled_tests_onnx_1_14_0(backend_test):
    # fails
    backend_test.exclude(r'test_averagepool_2d_dilations_cpu')
    backend_test.exclude(r'test_roialign_mode_max_cpu')

    # errors
    backend_test.exclude(r'test_constant_pad_negative_axes_cpu')
    backend_test.exclude(r'test_dequantizelinear_e4m3fn_cpu')
    backend_test.exclude(r'test_dequantizelinear_e5m2_cpu')
    backend_test.exclude(r'test_equal_string_broadcast_cpu')
    backend_test.exclude(r'test_equal_string_cpu')
    backend_test.exclude(r'test_quantizelinear_e4m3fn_cpu')
    backend_test.exclude(r'test_quantizelinear_e5m2_cpu')

    # The following tests fail due to the CastLike operator being unsupported
    backend_test.exclude(r'test_softplus_example_expanded_ver18_cpu')
    backend_test.exclude(r'test_softplus_expanded_ver18_cpu')


def create_backend_test(testname=None, target_device=None):
    if target_device is not None:
        c2.set_device(target_device)
    backend_test = MIGraphXBackendTest(c2, __name__)

    if testname:
        backend_test.include(testname + '.*')
    else:
        # Include all of the nodes that we support.
        # Onnx native node tests
        backend_test.include(r'.*test_abs.*')
        backend_test.include(r'.*test_acos.*')
        backend_test.include(r'.*test_acosh.*')
        backend_test.include(r'.*test_add.*')
        backend_test.include(r'.*test_and.*')
        backend_test.include(r'.*test_argmax.*')
        backend_test.include(r'.*test_argmin.*')
        backend_test.include(r'.*test_asin.*')
        backend_test.include(r'.*test_asinh.*')
        backend_test.include(r'.*test_atan.*')
        backend_test.include(r'.*test_atanh.*')
        backend_test.include(r'.*test_averagepool.*')
        backend_test.include(r'.*test_AvgPool.*')
        backend_test.include(r'.*test_BatchNorm.*eval.*')
        backend_test.include(r'.*test_ceil.*')
        backend_test.include(r'.*test_celu.*')
        backend_test.include(r'.*test_clip.*')
        backend_test.include(r'.*test_concat.*')
        backend_test.include(r'.*test_constant.*')
        backend_test.include(r'.*test_Conv[1-3]d*')
        backend_test.include(r'.*test_cos.*')
        backend_test.include(r'.*test_cosh.*')
        backend_test.include(r'.*test_depthtospace.*')
        backend_test.include(r'.*test_dequantizelinear')
        backend_test.include(r'.*test_div.*')
        backend_test.include(r'.*test_dropout.*')
        backend_test.include(r'.*test_ELU*')
        backend_test.include(r'.*test_elu.*')
        backend_test.include(r'.*test_equal.*')
        backend_test.include(r'.*test_Embedding*')
        backend_test.include(r'.*test_exp.*')
        backend_test.include(r'.*test_eyelike.*')
        backend_test.include(r'.*test_flatten.*')
        backend_test.include(r'.*test_floor.*')
        backend_test.include(r'.*test_fmod.*')
        backend_test.include(r'.*test_gather.*')
        backend_test.include(r'.*test_gemm.*')
        backend_test.include(r'.*test_globalaveragepool.*')
        backend_test.include(r'.*test_globalmaxpool.*')
        backend_test.include(r'.*test_greater.*')
        backend_test.include(r'.*test_hardsigmoid.*')
        backend_test.include(r'.*test_hardswish.*')
        backend_test.include(r'.*test_identity.*')
        backend_test.include(r'.*test_if.*')
        backend_test.include(r'.*test_isnan.*')
        backend_test.include(r'.*test_LeakyReLU*')
        backend_test.include(r'.*test_leakyrelu.*')
        backend_test.include(r'.*test_less.*')
        backend_test.include(r'.*test_Linear.*')
        backend_test.include(r'.*test_log.*')
        backend_test.include(r'.*test_logsoftmax.*')
        backend_test.include(r'.*test_LogSoftmax.*')
        backend_test.include(r'.*test_log_softmax.*')
        backend_test.include(r'.*test_lrn.*')
        backend_test.include(r'.*test_matmul.*')
        backend_test.include(r'.*test_max.*')
        backend_test.include(r'.*test_MaxPool[1-9]d.*')
        backend_test.include(r'.*test_mean.*')
        backend_test.include(r'.*test_min.*')
        backend_test.include(r' .*test_mod.*')
        backend_test.include(r'.*test_mul.*')
        backend_test.include(r'.*test_multinomial.*')
        backend_test.include(r'.*test_Multinomial.*')
        backend_test.include(r'.*test_mvn.*')
        backend_test.include(r'.*test_neg.*')
        backend_test.include(r'.*test_not.*')
        backend_test.include(r'.*test_operator_addmm.*')
        backend_test.include(r'.*test_operator_basic.*')
        backend_test.include(r'.*test_operator_chunk.*')
        backend_test.include(r'.*test_operator_clip.*')
        backend_test.include(r'.*test_operator_concat2.*')
        backend_test.include(r'.*test_operator_conv_.*')
        backend_test.include(r'.*test_operator_exp.*')
        backend_test.include(r'.*test_operator_flatten.*')
        backend_test.include(r'.*test_operator_index.*')
        backend_test.include(r'.*test_operator_max_.*')
        backend_test.include(r'.*test_operator_maxpool.*')
        backend_test.include(r'.*test_operator_min.*')
        backend_test.include(r'.*test_operator_mod.*')
        backend_test.include(r'.*test_operator_mm.*')
        backend_test.include(r'.*test_operator_non_float_params.*')
        backend_test.include(r'.*test_operator_params.*')
        backend_test.include(r'.*test_operator_permute2.*')
        backend_test.include(r'.*test_operator_pow.*')
        backend_test.include(r'.*test_operator_reduced_mean_.*')
        backend_test.include(r'.*test_operator_reduced_mean_keepdim.*')
        backend_test.include(r'.*test_operator_reduced_sum_.*')
        backend_test.include(r'.*test_operator_reduced_sum_keepdim.*')
        backend_test.include(r'.*test_operator_selu.*')
        backend_test.include(r'.*test_operator_sqrt.*')
        backend_test.include(r'.*test_operator_symbolic_override.*')
        backend_test.include(r'.*test_operator_symbolic_override_nested.*')
        backend_test.include(r'.*test_operator_view.*')
        backend_test.include(r'.*test_or.*')
        backend_test.include(r'.*test_pow.*')
        backend_test.include(r'.*test_PoissonNLLLLoss_no_reduce*')
        backend_test.include(r'.*test_quantizelinear')
        backend_test.include(r'.*test_reciprocal.*')
        backend_test.include(r'.*test_reduce.*')
        backend_test.include(r'.*test_ReLU*')
        backend_test.include(r'.*test_relu.*')
        #backend_test.include(r'.*test_reversesequence.*')
        backend_test.include(r'.*test_RoiAlign*')
        backend_test.include(r'.*test_roialign.*')
        backend_test.include(r'.*test_scatter.*')
        backend_test.include(r'.*test_Scatter.*')
        backend_test.include(r'.*test_selu.*')
        backend_test.include(r'.*test_shape.*')
        backend_test.include(r'.*test_Sigmoid*')
        backend_test.include(r'.*test_sigmoid.*')
        backend_test.include(r'.*test_sin.*')
        backend_test.include(r'.*test_sinh.*')
        backend_test.include(r'.*test_size.*')
        backend_test.include(r'.*test_Softmax*')
        backend_test.include(r'.*test_softmax.*')
        backend_test.include(r'.*test_Softmin*')
        backend_test.include(r'.*test_Softplus*')
        backend_test.include(r'.*test_softplus.*')
        backend_test.include(r'.*test_softsign.*')
        backend_test.include(r'.*test_sqrt.*')
        backend_test.include(r'.*test_squeeze_cuda')
        backend_test.include(r'.*test_sub.*')
        backend_test.include(r'.*test_sum.*')
        backend_test.include(r'.*test_tan.*')
        backend_test.include(r'.*test_Tanh*')
        backend_test.include(r'.*test_tanh.*')
        backend_test.include(r'.*test_thresholdedrelu.*')
        backend_test.include(r'.*test_topk.*')
        backend_test.include(r'.*test_Topk.*')
        backend_test.include(r'.*test_transpose.*')
        backend_test.include(r'.*test_unsqueeze.*')
        backend_test.include(r'.*test_where*')
        backend_test.include(r'.*test_where.*')
        backend_test.include(r'.*test_xor.*')
        backend_test.include(r'.*test_ZeroPad2d*')

        # # Onnx native model tests
        backend_test.include(r'.*test_bvlc_alexnet.*')
        backend_test.include(r'.*test_densenet121.*')
        backend_test.include(r'.*test_inception_v1.*')
        backend_test.include(r'.*test_inception_v2.*')
        backend_test.include(r'.*test_resnet50.*')
        backend_test.include(r'.*test_shufflenet.*')
        backend_test.include(r'.*test_squeezenet.*')
        backend_test.include(r'.*test_vgg19.*')
        backend_test.include(r'.*test_zfnet512.*')

        # exclude unenabled ops get pulled in with wildcards
        # test_constant_pad gets pulled in with the test_constant* wildcard. Explicitly disable padding tests for now.
        # Operator MATMULINTEGER is not supported by TRT
        backend_test.exclude(r'.*test_matmulinteger.*')
        backend_test.exclude(r'.*test_maxunpool.*')
        # Absolute diff failed because
        # numpy compares the difference between actual and desired to atol + rtol * abs(desired)

        # failed test cases
        backend_test.exclude(
            r'test_argmax_keepdims_example_select_last_index_cpu')
        backend_test.exclude(
            r'test_argmax_negative_axis_keepdims_example_select_last_index_cpu'
        )
        backend_test.exclude(
            r'test_argmax_no_keepdims_example_select_last_index_cpu')
        backend_test.exclude(
            r'test_argmin_keepdims_example_select_last_index_cpu')
        backend_test.exclude(
            r'test_argmin_negative_axis_keepdims_example_select_last_index_cpu'
        )
        backend_test.exclude(
            r'test_argmin_no_keepdims_example_select_last_index_cpu')
        backend_test.exclude(r'test_lrn_cpu')
        backend_test.exclude(r'test_lrn_default_cpu')
        backend_test.exclude(r'test_maxpool_2d_dilations_cpu')
        backend_test.exclude(r'test_MaxPool2d_stride_padding_dilation_cpu')
        backend_test.exclude(r'test_MaxPool1d_stride_padding_dilation_cpu')
        backend_test.exclude(
            r'test_maxpool_with_argmax_2d_precomputed_pads_cpu')
        backend_test.exclude(
            r'test_maxpool_with_argmax_2d_precomputed_strides_cpu')

        # error cases
        backend_test.exclude(r'test_constant_pad_cpu')
        backend_test.exclude(r'test_constantofshape_float_ones_cpu')
        backend_test.exclude(r'test_constantofshape_int_shape_zero_cpu')
        backend_test.exclude(r'test_constantofshape_int_zeros_cpu')
        backend_test.exclude(r'test_expand_dim_changed_cpu')
        backend_test.exclude(r'test_expand_dim_unchanged_cpu')
        backend_test.exclude(r'test_expand_shape_model1_cpu')
        backend_test.exclude(r'test_expand_shape_model2_cpu')
        backend_test.exclude(r'test_expand_shape_model3_cpu')
        backend_test.exclude(r'test_expand_shape_model4_cpu')
        backend_test.exclude(r'test_identity_sequence_cpu')
        backend_test.exclude(r'test_maxpool_2d_uint8_cpu')
        backend_test.exclude(r'test_negative_log_likelihood_loss_*')

        # all reduce ops have dynamic axes inputs
        backend_test.exclude(r'test_softmax_cross_entropy_*')
        backend_test.exclude(r'test_Embedding_cpu')

        # real model tests
        backend_test.exclude(r'test_inception_v1_cpu')
        backend_test.exclude(r'test_resnet50_cpu')
        backend_test.exclude(r'test_squeezenet_cpu')

        # additional cases disabled for a specific onnx version
        if version.parse(onnx.__version__) <= version.parse("1.7.0"):
            disabled_tests_onnx_1_7_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.8.0"):
            disabled_tests_onnx_1_8_1(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.10.0"):
            disabled_tests_onnx_1_10_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.11.0"):
            disabled_tests_onnx_1_11_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.12.0"):
            disabled_tests_onnx_1_12_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.13.0"):
            disabled_tests_onnx_1_13_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.14.0"):
            disabled_tests_onnx_1_14_0(backend_test)


# import all test cases at global scope to make
# them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)

    return backend_test


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='Run the ONNX backend tests using MIGraphX.')

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual
    # test case names.
    parser.add_argument(
        '-t',
        '--test-name',
        dest='testname',
        type=str,
        help=
        "Only run tests that match this value. Matching is regex based, and '.*' is automatically appended"
    )
    parser.add_argument('-d',
                        '--device',
                        dest='device',
                        type=str,
                        help="Specify the device to run test on")

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if args.device is not None:
        print("run on {} device....".format(args.device))
    else:
        print("Default GPU device is used ....")

    return args


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        sys.exit()

    args = parse_args()
    backend_test = create_backend_test(args.testname, args.device)
    unittest.main()

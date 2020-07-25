import sys
if sys.version_info < (3, 0):
    sys.exit()

import argparse
import json
import os
import platform
import unittest
import onnx
import onnx.backend.test
import numpy as np
from onnx_migraphx.backend import MIGraphXBackend as c2

pytest_plugins = 'onnx.backend.test.report',


class MIGraphXBackendTest(onnx.backend.test.BackendTest):
    def __init__(self, backend, parent_module=None):
        super(MIGraphXBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype, outputs[i].dtype)
            if ref_outputs[i].dtype == np.object:
                np.testing.assert_array_equal(ref_outputs[i], outputs[i])
            else:
                np.testing.assert_allclose(ref_outputs[i],
                                           outputs[i],
                                           rtol=1e-3,
                                           atol=1e-5)


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
        backend_test.include(r'.*test_clip.*')
        backend_test.include(r'.*test_concat.*')
        backend_test.include(r'.*test_constant.*')
        backend_test.include(r'.*test_Conv[1-3]d*')
        backend_test.include(r'.*test_cos.*')
        backend_test.include(r'.*test_cosh.*')
        backend_test.include(r'.*test_depthtospace.*')
        backend_test.include(r'.*test_div.*')
        backend_test.include(r'.*test_dropout.*')
        backend_test.include(r'.*test_ELU*')
        backend_test.include(r'.*test_elu.*')
        backend_test.include(r'.*test_equal.*')
        backend_test.include(r'.*test_Embedding*')
        backend_test.include(r'.*test_exp.*')
        backend_test.include(r'.*test_flatten.*')
        backend_test.include(r'.*test_floor.*')
        backend_test.include(r'.*test_gather.*')
        backend_test.include(r'.*test_gemm.*')
        backend_test.include(r'.*test_globalaveragepool.*')
        backend_test.include(r'.*test_globalmaxpool.*')
        backend_test.include(r'.*test_greater.*')
        backend_test.include(r'.*test_hardsigmoid.*')
        backend_test.include(r'.*test_identity.*')
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
        backend_test.include(r'.*test_mul.*')
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
        backend_test.include(r'.*test_pow.*')
        backend_test.include(r'.*test_PoissonNLLLLoss_no_reduce*')
        backend_test.include(r'.*test_reciprocal.*')
        backend_test.include(r'.*test_reduce.*')
        backend_test.include(r'.*test_ReLU*')
        backend_test.include(r'.*test_relu.*')
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
        backend_test.include(r'.*test_transpose.*')
        backend_test.include(r'.*test_unsqueeze.*')
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
        backend_test.exclude(r'test_dropout_default_mask_cpu')
        backend_test.exclude(r'test_dropout_default_mask_ratio_cpu')
        backend_test.exclude(r'test_logsoftmax_axis_0_cpu')
        backend_test.exclude(r'test_logsoftmax_axis_1_cpu')
        backend_test.exclude(r'test_logsoftmax_default_axis_cpu')
        backend_test.exclude(r'test_lrn_cpu')
        backend_test.exclude(r'test_lrn_default_cpu')
        backend_test.exclude(r'test_maxpool_2d_dilations_cpu')
        backend_test.exclude(
            r'test_maxpool_with_argmax_2d_precomputed_pads_cpu')
        backend_test.exclude(
            r'test_maxpool_with_argmax_2d_precomputed_strides_cpu')
        backend_test.exclude(r'test_softmax_axis_0_cpu')
        backend_test.exclude(r'test_softmax_axis_1_cpu')
        backend_test.exclude(r'test_softmax_default_axis_cpu')

        # error cases
        backend_test.exclude(r'test_averagepool_2d_ceil_cpu')
        backend_test.exclude(r'test_clip_default_inbounds_cpu')
        backend_test.exclude(r'test_clip_default_int8_inbounds_cpu')
        backend_test.exclude(r'test_clip_default_int8_max_cpu')
        backend_test.exclude(r'test_clip_default_max_cpu')
        backend_test.exclude(r'test_constant_pad_cpu')
        backend_test.exclude(r'test_constantofshape_float_ones_cpu')
        backend_test.exclude(r'test_constantofshape_int_shape_zero_cpu')
        backend_test.exclude(r'test_constantofshape_int_zeros_cpu')
        backend_test.exclude(r'test_depthtospace_crd_mode_cpu')
        backend_test.exclude(r'test_depthtospace_crd_mode_example_cpu')
        backend_test.exclude(r'test_depthtospace_dcr_mode_cpu')
        backend_test.exclude(r'test_depthtospace_example_cpu')
        backend_test.exclude(r'test_equal_bcast_cpu')
        backend_test.exclude(r'test_equal_cpu')
        backend_test.exclude(r'test_expand_dim_changed_cpu')
        backend_test.exclude(r'test_expand_dim_unchanged_cpu')
        backend_test.exclude(r'test_gather_0_cpu')
        backend_test.exclude(r'test_gather_1_cpu')
        backend_test.exclude(r'test_gather_elements_0_cpu')
        backend_test.exclude(r'test_gather_elements_1_cpu')
        backend_test.exclude(r'test_gather_elements_negative_indices_cpu')
        backend_test.exclude(r'test_gather_negative_indices_cpu')
        backend_test.exclude(r'test_gathernd_example_float32_cpu')
        backend_test.exclude(r'test_gathernd_example_int32_batch_dim1_cpu')
        backend_test.exclude(r'test_gathernd_example_int32_cpu')
        backend_test.exclude(r'test_greater_bcast_cpu')
        backend_test.exclude(r'test_greater_cpu')
        backend_test.exclude(r'test_greater_equal_bcast_cpu')
        backend_test.exclude(r'test_greater_equal_bcast_expanded_cpu')
        backend_test.exclude(r'test_greater_equal_cpu')
        backend_test.exclude(r'test_greater_equal_expanded_cpu')
        backend_test.exclude(r'test_hardsigmoid_cpu')
        backend_test.exclude(r'test_hardsigmoid_default_cpu')
        backend_test.exclude(r'test_hardsigmoid_example_cpu')
        backend_test.exclude(r'test_less_bcast_cpu')
        backend_test.exclude(r'test_less_cpu')
        backend_test.exclude(r'test_less_equal_bcast_cpu')
        backend_test.exclude(r'test_less_equal_bcast_expanded_cpu')
        backend_test.exclude(r'test_less_equal_cpu')
        backend_test.exclude(r'test_less_equal_expanded_cpu')
        backend_test.exclude(r'test_max_float16_cpu')
        backend_test.exclude(r'test_max_int64_cpu')
        backend_test.exclude(r'test_max_uint64_cpu')
        backend_test.exclude(r'test_maxpool_2d_ceil_cpu')
        backend_test.exclude(r'test_maxpool_2d_uint8_cpu')
        backend_test.exclude(r'test_mean_example_cpu')
        backend_test.exclude(r'test_mean_one_input_cpu')
        backend_test.exclude(r'test_mean_two_inputs_cpu')
        backend_test.exclude(r'test_min_float16_cpu')
        backend_test.exclude(r'test_min_int64_cpu')
        backend_test.exclude(r'test_min_uint64_cpu')
        backend_test.exclude(r'test_negative_log_likelihood_loss_*')
        backend_test.exclude(r'test_not_2d_cpu')
        backend_test.exclude(r'test_not_3d_cpu')
        backend_test.exclude(r'test_not_4d_cpu')
        backend_test.exclude(r'test_pow_types_*')
        backend_test.exclude(r'test_selu_cpu')
        backend_test.exclude(r'test_selu_default_cpu')
        backend_test.exclude(r'test_selu_example_cpu')
        backend_test.exclude(r'test_size_cpu')
        backend_test.exclude(r'test_size_example_cpu')
        backend_test.exclude(r'test_softmax_cross_entropy_*')
        backend_test.exclude(r'test_softplus_cpu')
        backend_test.exclude(r'test_softplus_example_cpu')
        backend_test.exclude(r'test_softsign_cpu')
        backend_test.exclude(r'test_softsign_example_cpu')
        backend_test.exclude(r'test_thresholdedrelu_cpu')
        backend_test.exclude(r'test_thresholdedrelu_default_cpu')
        backend_test.exclude(r'test_thresholdedrelu_example_cpu')
        backend_test.exclude(r'test_Embedding_cpu')
        backend_test.exclude(r'test_Embedding_sparse_cpu')
        backend_test.exclude(r'test_Softplus_cpu')
        backend_test.exclude(r'test_operator_non_float_params_cpu')
        backend_test.exclude(r'test_operator_selu_cpu')
        backend_test.exclude(r'test_expand_shape_model1_cpu')
        backend_test.exclude(r'test_expand_shape_model2_cpu')
        backend_test.exclude(r'test_expand_shape_model3_cpu')
        backend_test.exclude(r'test_expand_shape_model4_cpu')

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

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
import argparse
import numpy as np
import migraphx
import onnxruntime as ort
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'MIGraphX accuracy checker for distilgpt2_fp16. Used to verify MIGraphX\'s output \
                                                  is within tolerance of onnx runtime\'s expected output.'
    )
    parser.add_argument('model', help='path to distilgpt2_fp16.onnx')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
    parser.add_argument('--atol',
                        type=float,
                        default=1e-3,
                        help='absolute tolerance (default = 1e-3)')
    parser.add_argument('--rtol',
                        type=float,
                        default=1e-3,
                        help='relative tolerance (default = 1e-3)')
    parser.add_argument('--target',
                        type=str,
                        default='gpu',
                        help='target to compile and run MIGraphX on')
    parser.add_argument('--provider',
                        type=str,
                        default='CPUExecutionProvider',
                        help='execution provider for onnx runtime \
                                (default = CPUExecutionProvider)')
    parser.add_argument('--ort-logging',
                        dest="ort_logging",
                        action='store_true',
                        default=False,
                        help='Turn on ort VERBOSE logging via session options')

    parser.add_argument(
        '--disable-offload-copy',
        dest="offload_copy",
        action='store_false',
        default=True,
        help=
        'Disable offload copying (user must handle copy to and from device)')

    parser.add_argument(
        '--disable-fast-math',
        dest="fast_math",
        action='store_false',
        default=True,
        help='Disable fast math optimizations (etc: rewrite_gelu)')

    parser.add_argument('--exhaustive_tune',
                        dest="exhaustive_tune",
                        action='store_true',
                        default=False,
                        help='Enable exhaustive tuning for solutions')

    args = parser.parse_args()

    return args, parser


def check_correctness(gold_outputs,
                      outputs,
                      rtol,
                      atol,
                      verbose=False):
    if len(gold_outputs) != len(outputs):
        print(f'Number of outputs {len(outputs)} is not equal to\
                expected number {len(gold_outputs)}')
        return False

    out_num = len(gold_outputs)
    ret = True

    for i in range(out_num):
        if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
            ret = False
            if verbose:
                with np.printoptions(threshold=np.inf):
                    print(f'\nOutput {i} is incorrect ...')
                    print(f'Expected value: \n{gold_outputs[i]}')
                    print('\n......\n')
                    print(f'Actual value: \n{outputs[i]}\n')
                    diff = gold_outputs[i] - outputs[i]
                    max_diff = np.max(np.abs(diff))
                    print(f'Max Difference: {max_diff}')
            else:
                print('Outputs do not match')
                break
    return ret


def get_np_datatype(in_type):
    datatypes = {
        'double_type': np.float64,
        'float_type': np.float32,
        'half_type': np.half,
        'int64_type': np.int64,
        'uint64_type': np.uint64,
        'int32_type': np.int32,
        'uint32_type': np.uint32,
        'int16_type': np.int16,
        'uint16_type': np.uint16,
        'int8_type': np.int8,
        'uint8_type': np.uint8,
        'bool_type': bool
    }
    return datatypes[in_type]


def main():
    args, parser = parse_args()

    model_name = args.model

    input_dims = {'input_ids': [1, 347]}

    model = migraphx.parse_onnx(model_name, map_input_dims=input_dims)

    if args.verbose:
        print(model)

    model.compile(
        migraphx.get_target(args.target),
        offload_copy=args.offload_copy,
        fast_math=args.fast_math,
        exhaustive_tune=args.exhaustive_tune,
    )

    params = {}
    test_inputs = {}
    input_ids = np.array([
      3855,  3608,   287,    12,  6057, 33849,   351,   716,    72, 26762,
     18199,     0,  2329,  9814,   284,  4776,   649,  3435,    11,   983,
     12881,    11,   393,   584, 28429,    13,  1881,   716,    72, 26762,
       743,   670,   351,  3294,  1830,    13,   921,  1244,   651,   649,
     27655,    11,  1176,    12,  4739,    11,   393,   584,  1257, 17568,
        13,  7502,   318,   262,  1388,  2095,   287,   383,  9883,   286,
     22166,  1830,    13,   317,  1862,  2933,  2877,   287,  6707, 25135,
        11,  7502,   318,  1690,  1813,   262,  4876,   286, 48329,  8449,
     22166,   290,  6707, 25135,   422,   262, 13573, 12003, 25906, 23207,
       623, 24263,    13,   367, 10344,   284,   262,   886,    11,  7502,
       318,  1900,   407,  6974,   355,   257,  4293,   475,   355,   257,
      6194,   286, 11917,    11,  4202,   290, 11501,   355,   880,    13,
      3082, 16873,  5776,    25, 19430,   290,  4149,    25,  3115, 18214,
     14266,    13,   329, 16591,   471,  4149,  5514,    25,   383,  9883,
       286, 22166,    25, 19715,  8449,  5572, 10682, 32872,   807,  6707,
     25135, 12090,  8599, 42616,    25, 20215, 26885, 40488,  9440, 47379,
      7461, 15590, 22776, 14843,  1343, 10682,  3615,   838,   716,    72,
     26762,  9814,    25,  9714,   338, 33575, 44733,  3115, 10682, 21521,
       609, 27567,    12, 14350,    78,     0, 38636, 47137, 38936,   338,
     22173,  5098,  2159,  9678, 31913,   416, 24148, 18878,   357, 32348,
        18,  5258,     8,  9678, 31913,   416, 24148, 18878,   357,    54,
      4178,    52,     8,  8175,    74, 35942,  9595,  6707, 25135, 12090,
     14270, 45430,  5537,    25, 11937,  5221, 45430,  5537,    25, 20641,
      6932, 34588, 17738,  8858,  7670,    25, 30958, 19530, 10682,  1222,
     18426,   379,   262, 15338,  1584, 11514,  5776, 23965,    25, 11397,
      3851,   672,   313, 12558, 10682,  1222, 14213,   716,    72, 26762,
     13879,   357,    54,  4178,    52,     8, 12558, 10682,  1222, 14213,
       716,    72, 26762, 13879,   357, 32348,    18,  5258,     8, 15085,
      1214,   513,    35, 10485,   362,   383,  9883,   286, 22166,    25,
     26988,   286,   262,  6183, 41778, 29658,  1222, 38936,   338, 22173,
      5098,  2159, 13792, 31910,    25,   968, 14697,   532, 19134,   716,
        72, 26762, 13792, 31910,    25,   968, 14697, 10682,  3615,  2907,
     13063, 10682,  3615,  2907, 13063,   532,  3615, 22358, 10682,  7092,
      3115, 30783, 13756,   270, 24464,  3764, 40254, 45547,    25, 18037,
       286, 17284,   544, 10682, 32872,   807, 21907]).reshape([1, 347])
    test_inputs['input_ids'] = input_ids
    migraphx_arg = migraphx.argument(input_ids)
    if not args.offload_copy:
        migraphx_arg = migraphx.to_gpu(migraphx_arg)
    params['input_ids'] = migraphx_arg

    if not args.offload_copy:
        pred_migx = np.array(migraphx.from_gpu(model.run(params)[-1]))
    else:
        pred_migx = np.array(model.run(params)[-1])

    sess_op = ort.SessionOptions()
    if args.ort_logging:
        sess_op.log_verbosity_level = 0
        sess_op.log_severity_level = 0
    sess = ort.InferenceSession(model_name,
                                sess_options=sess_op,
                                providers=[args.provider])
    ort_params = {}
    for i in sess.get_inputs():
        ort_params[i.name] = test_inputs[i.name]
    try:
        pred_fw = sess.run(None, ort_params)[-1]
    except Exception as e:
        raise e

    is_correct = check_correctness(pred_fw, pred_migx, args.atol,
                                   args.rtol, args.verbose)
    verbose_string = ' Rerun with --verbose for detailed information.' \
            if not args.verbose else ''
    if is_correct:
        print('PASSED: MIGraphX meets tolerance')
    else:
        print('FAILED: MIGraphX is not within tolerance.' + verbose_string)


if __name__ == '__main__':
    main()

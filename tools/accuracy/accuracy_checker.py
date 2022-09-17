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
import argparse
import numpy as np
import migraphx
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'MIGraphX accuracy checker. Use to verify onnx files to ensure MIGraphX\'s output \
                                                  is within tolerance of onnx runtime\'s expected output.'
    )
    req_args = parser.add_argument_group(title='required arguments')
    req_args.add_argument('--onnx',
                          type=str,
                          required=True,
                          help='path to onnx file')
    req_args.add_argument('--provider',
                          type=str,
                          default='CPUExecutionProvider',
                          help='execution provider for onnx runtime \
                                (default = CPUExecutionProvider)')
    parser.add_argument('--batch',
                        type=int,
                        default=1,
                        help='batch size (if specified in onnx file)')
    parser.add_argument('--fill1',
                        action='store_true',
                        help='fill all arguments with a value of 1')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
    parser.add_argument('--tolerance',
                        type=float,
                        default=1e-3,
                        help='accuracy tolerance (default = 1e-3)')
    args = parser.parse_args()

    return args


# taken from ../test_runner.py
def check_correctness(gold_outputs,
                      outputs,
                      rtol=1e-3,
                      atol=1e-3,
                      verbose=False):
    if len(gold_outputs) != len(outputs):
        print('Number of outputs {} is not equal to expected number {}'.format(
            len(outputs), len(gold_outputs)))
        return False

    out_num = len(gold_outputs)
    ret = True
    for i in range(out_num):
        if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
            ret = False
            if verbose:
                print('\nOutput {} is incorrect ...'.format(i))
                print('Expected value: \n{}'.format(gold_outputs[i]))
                print('......')
                print('Actual value: \n{}\n'.format(outputs[i]))
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
        'bool_type': np.bool_
    }
    return datatypes[in_type]


def main():
    args = parse_args()

    model_name = args.onnx
    batch = args.batch

    model = migraphx.parse_onnx(model_name, default_dim_value=batch)

    model.compile(migraphx.get_target('gpu'), offload_copy=False)

    params = {}
    test_inputs = {}
    for name, shape in model.get_parameter_shapes().items():
        if args.verbose:
            print('Parameter {} -> {}'.format(name, shape))
        in_shape = shape.lens()
        in_type = shape.type_string()
        if not args.fill1:
            test_input = np.random.rand(*(in_shape)).astype(
                get_np_datatype(in_type))
        else:
            test_input = np.ones(in_shape).astype(get_np_datatype(in_type))
        test_inputs[name] = test_input
        params[name] = migraphx.to_gpu(migraphx.argument(test_input))

    pred_migx = np.array(migraphx.from_gpu(model.run(params)[-1]))

    sess = ort.InferenceSession(model_name, providers=[args.provider])

    ort_params = {}
    for input in sess.get_inputs():
        ort_params[input.name] = test_inputs[input.name]

    pred_ort = sess.run(None, ort_params)[-1]

    is_correct = check_correctness(pred_ort, pred_migx, args.tolerance,
                                   args.tolerance, args.verbose)
    verbose_string = ' Rerun with --verbose for detailed information.' \
            if not args.verbose else ''
    if is_correct:
        print('PASSED: MIGraphX meets tolerance')
    else:
        print('FAILED: MIGraphX is not within tolerance.' + verbose_string)


if __name__ == '__main__':
    main()

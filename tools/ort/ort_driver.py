#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
import onnxruntime as ort
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'ONNX Runtime python driver. Use to run onnx models using ONNX Runtime'
    )
    file_args = parser.add_argument_group(title='file type arguments')
    file_args.add_argument('--onnx', type=str, help='path to onnx file')
    parser.add_argument('--provider',
                        type=str,
                        default='CPUExecutionProvider',
                        help='execution provider for onnx runtime \
                                (default = CPUExecutionProvider)')
    parser.add_argument('--fill1',
                        action='store_true',
                        help='fill all arguments with a value of 1')
    parser.add_argument('--fill0',
                        action='store_true',
                        help='fill all arguments with a value of 0')
    parser.add_argument(
        '--default-dim-value',
        type=int,
        default=1,
        help='default dim value (if any specified in onnx file)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
    parser.add_argument('--input-dim',
                        type=str,
                        action='append',
                        help='specify input parameter dimension \
                                with the following format --input-dim input_name:dim0,dim1,dim2...'
                        )

    parser.add_argument('--ort-logging',
                        dest="ort_logging",
                        action='store_true',
                        default=False,
                        help='Turn on ort VERBOSE logging via session options')

    args = parser.parse_args()

    return args, parser


def get_np_datatype(in_type):
    datatypes = {
        'tensor(double)': np.float64,
        'tensor(float)': np.float32,
        'tensor(float16)': np.half,
        'tensor(int64)': np.int64,
        'tensor(uint64)': np.uint64,
        'tensor(int32)': np.int32,
        'tensor(uint32)': np.uint32,
        'tensor(int16)': np.int16,
        'tensor(uint16)': np.uint16,
        'tensor(int8)': np.int8,
        'tensor(uint8)': np.uint8,
    }
    return datatypes[in_type]


def main():
    args, parser = parse_args()

    if args.onnx == None:
        print('Error: please specify an onnx file')
        parser.print_help()
        sys.exit(-1)

    model_name = args.onnx

    custom_inputs = args.input_dim

    input_dims = {}
    if custom_inputs != None:
        for input in custom_inputs:
            input_dim = ''.join(input.split(':')[:-1])
            dims = [int(dim) for dim in input.split(':')[-1].split(',')]
            input_dims[input_dim] = dims

    sess_op = ort.SessionOptions()

    if args.ort_logging:
        sess_op.log_verbosity_level = 0
        sess_op.log_severity_level = 0

    sess = ort.InferenceSession(model_name,
                                sess_options=sess_op,
                                providers=[args.provider])

    test_inputs = {}
    for input in sess.get_inputs():
        name = input.name
        in_shape = input.shape
        in_type = input.type

        for idx, dim in enumerate(in_shape):
            if not isinstance(dim, int):
                if args.verbose:
                    print(
                        f'''Dim param found at index {idx}: {dim}. '''
                        f'''Setting to default dim value of {args.default_dim_value}.'''
                    )
                in_shape[idx] = args.default_dim_value

        if name in input_dims:
            in_shape = input_dims[name]

        if args.verbose:
            print(f'Parameter {name} -> {in_shape}, {in_type}')

        if not args.fill1 and not args.fill0:
            test_input = np.random.rand(*(in_shape)).astype(
                get_np_datatype(in_type))
        elif not args.fill0:
            test_input = np.ones(in_shape).astype(get_np_datatype(in_type))
        else:
            test_input = np.zeros(in_shape).astype(get_np_datatype(in_type))
        test_inputs[name] = test_input

    ort_params = {}
    for input in sess.get_inputs():
        ort_params[input.name] = test_inputs[input.name]

    try:
        pred_ort = sess.run(None, ort_params)
        if args.verbose:
            for idx, output in enumerate(pred_ort):
                print(f'output at index {idx}: {output}')
    except Exception as e:
        if any(input_dims):
            print(
                'Error: custom input dim may not be compatible with onnx runtime'
            )
        raise e
    print('onnx runtime driver completed successfully.')


if __name__ == '__main__':
    main()

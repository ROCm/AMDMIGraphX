#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
import os
import sys
import argparse
import numpy as np
import onnx
from onnx import numpy_helper
import migraphx


def parse_args():
    parser = argparse.ArgumentParser(description="MIGraphX test runner")
    parser.add_argument('test_dir',
                        type=str,
                        metavar='test_loc',
                        help='folder where the test is stored')
    parser.add_argument('--target',
                        type=str,
                        default='gpu',
                        help='Specify where the tests execute (ref, gpu)')
    parser.add_argument('--fp16', action='store_true', help='Quantize to fp16')
    parser.add_argument('--atol',
                        type=float,
                        default=1e-3,
                        help='The absolute tolerance parameter')
    parser.add_argument('--rtol',
                        type=float,
                        default=1e-3,
                        help='The relative tolerance parameter')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
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

    return args


def get_sub_folders(dir_name):
    dir_contents = os.listdir(dir_name)
    folders = []
    for item in dir_contents:
        tmp_item = dir_name + '/' + item
        if os.path.isdir(tmp_item):
            folders.append(item)
    folders.sort()

    return folders


def get_test_cases(dir_name):
    return get_sub_folders(dir_name)


def get_model_name(dir_name):
    dir_contents = os.listdir(dir_name)
    for item in dir_contents:
        file_name = dir_name + '/' + item
        if os.path.isfile(file_name) and file_name.endswith('.onnx'):
            return item

    return ''


def read_pb_file(filename):
    with open(filename, 'rb') as pfile:
        data_str = pfile.read()
        tensor = onnx.TensorProto()
        tensor.ParseFromString(data_str)
        np_array = numpy_helper.to_array(tensor)

    return tensor.name, np_array


def wrapup_inputs(io_folder, param_names):
    param_map = {}
    data_array = []
    name_array = []
    for i in range(len(param_names)):
        file_name = io_folder + '/input_' + str(i) + '.pb'
        name, data = read_pb_file(file_name)
        param_map[name] = data
        data_array.append(data)
        if name:
            name_array.append(name)

    if len(name_array) < len(data_array):
        param_map = {}
        for i, param in enumerate(param_names):
            param_map[param] = data_array[i]

        return param_map

    for name in param_names:
        if not name in param_map:
            print(f"Input {name} does not exist!")
            sys.exit()

    return param_map


def read_outputs(io_folder, out_names):
    outputs = []
    data_array = []
    name_array = []
    for i in range(len(out_names)):
        file_name = io_folder + '/output_' + str(i) + '.pb'
        name, data = read_pb_file(file_name)
        data_array.append(data)
        if name:
            name_array.append(name)

    if len(name_array) < len(data_array):
        return data_array

    for name in out_names:
        index = name_array.index(name)
        outputs.append(data_array[index])

    return outputs


def model_parameter_names(model_file_name):
    with open(model_file_name, 'rb') as pfile:
        data_str = pfile.read()
        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(data_str)
        init_names = {i.name for i in model_proto.graph.initializer}
        param_names = [
            input.name for input in model_proto.graph.input
            if input.name not in init_names
        ]

        return param_names


def model_output_names(model_file_name):
    with open(model_file_name, 'rb') as pfile:
        data_str = pfile.read()
        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(data_str)
        output_names = [out.name for out in model_proto.graph.output]

        return output_names


def get_input_shapes(sample_case, param_names):
    param_shape_map = {}
    name_array = []
    shape_array = []
    for i in range(len(param_names)):
        file_name = sample_case + '/input_' + str(i) + '.pb'
        name, data = read_pb_file(file_name)
        param_shape_map[name] = data.shape
        shape_array.append(data.shape)
        if name:
            name_array.append(name)

    if len(name_array) < len(shape_array):
        param_shape_map = {}
        for i, param in enumerate(param_names):
            param_shape_map[param] = shape_array[i]

        return param_shape_map

    for name in param_names:
        if not name in param_shape_map:
            print("Input {} does not exist!".format(name))
            sys.exit()

    return param_shape_map


def run_one_case(model, param_map):
    # convert np array to model argument
    pp = {}
    for key, val in param_map.items():
        pp[key] = migraphx.argument(val)

    # run the model
    model_outputs = model.run(param_map)

    # convert argument to np array
    outputs = []
    for output in model_outputs:
        outputs.append(np.array(output))

    return outputs


def check_correctness(gold_outputs,
                      outputs,
                      rtol=1e-3,
                      atol=1e-3,
                      verbose=False):
    if len(gold_outputs) != len(outputs):
        print(
            f'Number of outputs {len(outputs)} is not equal to expected number {len(gold_outputs)}'
        )
        return False

    out_num = len(gold_outputs)
    ret = True

    for i in range(out_num):
        if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
            ret = False
            if verbose:
                with np.printoptions(threshold=np.inf):
                    print(f'\nOutput {i} is incorrect ...')
                    print(f'Expected value: \n{gold_outputs[i]}\n')
                    print('\n......\n')
                    print(f'Actual value: \n{outputs[i]}\n')
            else:
                print('Outputs do not match')
                break
    return ret


def tune_input_shape(model, input_data):
    param_shapes = model.get_parameter_shapes()
    input_shapes = {}
    for name, s in param_shapes.items():
        assert name in input_data
        data_shape = list(input_data[name].shape)
        if not np.array_equal(data_shape, s.lens()):
            input_shapes[name] = data_shape

    return input_shapes


def main():
    args = parse_args()
    test_loc = args.test_dir
    test_name = os.path.basename(os.path.normpath(test_loc))

    print("Running test \"{}\" on target \"{}\" ...\n".format(
        test_name, args.target))

    # get model full path
    model_name = get_model_name(test_loc)
    model_path_name = test_loc + '/' + model_name

    # get param names
    param_names = model_parameter_names(model_path_name)

    # get output names
    output_names = model_output_names(model_path_name)

    # get test cases
    cases = get_test_cases(test_loc)
    sample_case = test_loc + '/' + cases[0]
    param_shapes = get_input_shapes(sample_case, param_names)
    for name, dims in param_shapes.items():
        print("Input: {}, shape: {}".format(name, dims))
    print()

    # read and compile model
    model = migraphx.parse_onnx(model_path_name, map_input_dims=param_shapes)
    if args.fp16:
        migraphx.quantize_fp16(model)

    model.compile(
        migraphx.get_target(args.target),
        offload_copy=args.offload_copy,
        fast_math=args.fast_math,
        exhaustive_tune=args.exhaustive_tune,
    )

    # get test cases
    case_num = len(cases)
    correct_num = 0
    for case_name in cases:
        io_folder = test_loc + '/' + case_name
        input_data = wrapup_inputs(io_folder, param_names)
        gold_outputs = read_outputs(io_folder, output_names)

        # if input shape is different from model shape, reload and recompile
        # model
        input_shapes = tune_input_shape(model, input_data)
        if len(input_shapes) != 0:
            model = migraphx.parse_onnx(model_path_name,
                                        map_input_dims=input_shapes)
            model.compile(migraphx.get_target(args.target))

        # run the model and return outputs
        output_data = run_one_case(model, input_data)

        # check output correctness
        ret = check_correctness(gold_outputs,
                                output_data,
                                atol=args.atol,
                                rtol=args.rtol,
                                verbose=args.verbose)
        if ret:
            correct_num += 1

        output_str = "PASSED" if ret else "FAILED"
        print("\tCase {}: {}".format(case_name, output_str))

    print("\nTest \"{}\" has {} cases:".format(test_name, case_num))
    print("\t Passed: {}".format(correct_num))
    print("\t Failed: {}".format(case_num - correct_num))
    if case_num > correct_num:
        error_num = case_num - correct_num
        raise ValueError(str(error_num) + " cases failed!")


if __name__ == "__main__":
    main()

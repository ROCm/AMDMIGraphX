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
import os, sys
import glob
import numpy as np
import argparse
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
    args = parser.parse_args()

    return args


def get_sub_folders(dir_name):
    dir_contents = os.listdir(dir_name)
    folders = []
    for item in dir_contents:
        # Skip AppleDouble/pax metadata that some zoo archives ship
        if item.startswith('.') or item == 'PaxHeader':
            continue
        tmp_item = dir_name + '/' + item
        if os.path.isdir(tmp_item):
            folders.append(item)
    folders.sort()

    return folders


def get_test_cases(dir_name):
    return get_sub_folders(dir_name)


def get_model_name(dir_name):
    dir_contents = os.listdir(dir_name)
    for item in sorted(dir_contents):
        # Skip AppleDouble sidecar files like ._model.onnx
        if item.startswith('._'):
            continue
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

    # fall back to positional mapping (input_i.pb -> i-th model input)
    if len(name_array) < len(data_array) or any(
            name not in param_map for name in param_names):
        return {param_names[i]: data_array[i] for i in range(len(param_names))}

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

    # fall back to positional order when names are absent or do not match
    if len(name_array) < len(data_array) or any(
            name not in name_array for name in out_names):
        return data_array

    for name in out_names:
        outputs.append(data_array[name_array.index(name)])

    return outputs


def model_parameter_names(model_file_name):
    with open(model_file_name, 'rb') as pfile:
        data_str = pfile.read()
        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(data_str)
        init_names = set([(i.name) for i in model_proto.graph.initializer])
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

    # fall back to positional mapping when names are absent or do not match
    if len(name_array) < len(shape_array) or any(
            name not in param_shape_map for name in param_names):
        return {param_names[i]: shape_array[i] for i in range(len(param_names))}

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


def check_correctness(gold_outputs, outputs, rtol=1e-3, atol=1e-3):
    if len(gold_outputs) != len(outputs):
        print("Number of outputs {} is not equal to expected number {}".format(
            len(outputs), len(gold_outputs)))
        return False

    out_num = len(gold_outputs)
    ret = True
    for i in range(out_num):
        gold = np.asarray(gold_outputs[i])
        actual = np.asarray(outputs[i])
        if gold.shape != actual.shape:
            print("\nOutput {} shape mismatch: expected {}, got {}".format(
                i, gold.shape, actual.shape))
            ret = False
            continue
        if not np.allclose(gold, actual, rtol, atol):
            print("\nOutput {} is incorrect ...".format(i))
            print("Expected value: \n{}".format(gold))
            print("......")
            print("Actual value: \n{}\n".format(actual))
            ret = False

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


def load_npz_case(npz_path):
    # Legacy caffe2-era ONNX zoo test data: an .npz holding object arrays
    # 'inputs' and 'outputs', each with the tensors in model order.
    data = np.load(npz_path, allow_pickle=True, encoding='bytes')
    keys = list(getattr(data, 'files', []))
    if 'inputs' not in keys or 'outputs' not in keys:
        raise KeyError("{}: expected 'inputs' and 'outputs' arrays, found {}".format(
            os.path.basename(npz_path), keys))
    inputs = [np.asarray(x) for x in data['inputs']]
    outputs = [np.asarray(x) for x in data['outputs']]
    return inputs, outputs


def run_npz_cases(test_loc, model_path_name, param_names, npz_files, args):
    target = args.target
    test_name = os.path.basename(os.path.normpath(test_loc))
    cases = [load_npz_case(f) for f in npz_files]

    # Positional mapping: the i-th stored tensor feeds the i-th model input.
    param_shapes = {
        param_names[i]: cases[0][0][i].shape
        for i in range(len(param_names))
    }
    for name, dims in param_shapes.items():
        print("Input: {}, shape: {}".format(name, dims))
    print()

    model = migraphx.parse_onnx(model_path_name, map_input_dims=param_shapes)
    if args.fp16:
        migraphx.quantize_fp16(model)
    model.compile(migraphx.get_target(target))

    correct_num = 0
    for idx, (inputs, gold_outputs) in enumerate(cases):
        input_data = {param_names[i]: inputs[i] for i in range(len(param_names))}

        # if input shape is different from model shape, reload and recompile
        input_shapes = tune_input_shape(model, input_data)
        if not len(input_shapes) == 0:
            model = migraphx.parse_onnx(model_path_name, map_input_dims=input_shapes)
            model.compile(migraphx.get_target(target))

        output_data = run_one_case(model, input_data)
        ret = check_correctness(gold_outputs,
                                output_data,
                                atol=args.atol,
                                rtol=args.rtol)
        if ret:
            correct_num += 1
        print("\tCase {}: {}".format(idx, "PASSED" if ret else "FAILED"))

    case_num = len(cases)
    print("\nTest \"{}\" has {} cases:".format(test_name, case_num))
    print("\t Passed: {}".format(correct_num))
    print("\t Failed: {}".format(case_num - correct_num))
    if case_num > correct_num:
        print(str(case_num - correct_num) + " cases failed!")
        sys.exit(1)


def main():
    args = parse_args()
    test_loc = args.test_dir
    target = args.target

    test_name = os.path.basename(os.path.normpath(test_loc))

    print("Running test \"{}\" on target \"{}\" ...\n".format(
        test_name, target))

    # get model full path
    model_name = get_model_name(test_loc)
    model_path_name = test_loc + '/' + model_name

    # get param names
    param_names = model_parameter_names(model_path_name)

    # get output names
    output_names = model_output_names(model_path_name)

    # get test cases; fall back to legacy test_data_*.npz when there are no
    # test_data_set_*/ folders (older caffe2-era zoo archives)
    cases = get_test_cases(test_loc)
    if not cases:
        npz_files = sorted(glob.glob(test_loc + '/test_data_*.npz'))
        if npz_files:
            run_npz_cases(test_loc, model_path_name, param_names, npz_files, args)
            return
        print("No test_data_set_* or test_data_*.npz found in {}".format(test_loc))
        sys.exit(1)
    sample_case = test_loc + '/' + cases[0]
    param_shapes = get_input_shapes(sample_case, param_names)
    for name, dims in param_shapes.items():
        print("Input: {}, shape: {}".format(name, dims))
    print()

    # read and compile model
    model = migraphx.parse_onnx(model_path_name, map_input_dims=param_shapes)
    if args.fp16:
        migraphx.quantize_fp16(model)
    model.compile(migraphx.get_target(target))

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
        if not len(input_shapes) == 0:
            model = migraphx.parse_onnx(model_path_name,
                                        map_input_dims=input_shapes)
            model.compile(migraphx.get_target(target))

        # run the model and return outputs
        output_data = run_one_case(model, input_data)

        # check output correctness
        ret = check_correctness(gold_outputs,
                                output_data,
                                atol=args.atol,
                                rtol=args.rtol)
        if ret:
            correct_num += 1

        output_str = "PASSED" if ret else "FAILED"
        print("\tCase {}: {}".format(case_name, output_str))

    print("\nTest \"{}\" has {} cases:".format(test_name, case_num))
    print("\t Passed: {}".format(correct_num))
    print("\t Failed: {}".format(case_num - correct_num))
    if case_num > correct_num:
        error_num = case_num - correct_num
        print(str(error_num) + " cases failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

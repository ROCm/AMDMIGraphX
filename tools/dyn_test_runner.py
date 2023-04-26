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
import os
import sys
import numpy as np
import argparse
import onnx
from onnx import numpy_helper
import migraphx


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIGraphX dynamic test runner")
    parser.add_argument('test_dir',
                        type=str,
                        metavar='test_loc',
                        help='folder where the test is stored')
    parser.add_argument('--target',
                        type=str,
                        default='gpu',
                        help='Specify where the tests execute (ref, gpu)')
    parser.add_argument(
        '--default_dyn_dim_value',
        type=str,
        help=
        'default dynamic_dimension to use if model has dynamic shapes, ex: "{1, 4, 2}"'
    )
    parser.add_argument(
        '--map_dyn_input_dims',
        type=str,
        help=
        'json dict of map_dyn_input_dims to pass to onnx_options while parsing'
    )
    args = parser.parse_args()

    return args


def parse_dyn_dim_str(dim_str):
    # expecting string like: "{1, 4, 2}" or "{2, 4}"
    if dim_str is None:
        return migraphx.shape.dynamic_dimension(1, 1, 0)
    dim_str = dim_str.strip('{}()')
    dims = [int(x) for x in dim_str.split(', ')]
    if len(dims) == 3:
        return migraphx.shape.dynamic_dimension(dims[0], dims[1], dims[2])
    return migraphx.shape.dynamic_dimension(dims[0], dims[1])


def parse_dyn_dims_str(dds_str):
    # expecting string like "[{1, 4, 2}, {4, 4}, {4, 4}]"
    dyn_dims = []
    start_ind = 0
    dds_str = dds_str.strip('[]')
    for i, v in enumerate(dds_str):
        if v == '{':
            start_ind = i
        elif v == '}':
            dyn_dims.append(dds_str[start_ind:i + 1])
    return [parse_dyn_dim_str(dd) for dd in dyn_dims]


def parse_map_dyn_input_str(dict_str):
    # return {input_name: list<dynamic_dimension>}
    # expecting string like: `{"A": [{1, 4, 2}, {4, 4}, {4, 4}], "B": [{2, 4}, {2, 4}]}`
    start_ind = 0
    in_quotes = False
    keys = []
    dyn_dim_strs = []
    for i, v in enumerate(dict_str):
        if v == '"':
            if not in_quotes:
                start_ind = i
                in_quotes = True
            else:
                keys.append(dict_str[start_ind:i + 1])
                in_quotes = False
        elif v == '[' and not in_quotes:
            start_ind = i
        elif v == ']' and not in_quotes:
            dyn_dim_strs.append(dict_str[start_ind:i + 1])
    dd_dict = {}
    for key, dds in zip(keys, dyn_dim_strs):
        x = parse_dyn_dims_str(dds)
        dd_dict[key.strip('"')] = x
    return dd_dict


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
        for i in range(len(param_names)):
            param_map[param_names[i]] = data_array[i]

        return param_map

    for name in param_names:
        if not name in param_map.keys():
            print("Input {} does not exist!".format(name))
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

    if len(name_array) < len(shape_array):
        param_shape_map = {}
        for i in range(len(param_names)):
            param_shape_map[param_names[i]] = shape_array[i]

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
        pp[key] = migraphx.to_gpu(migraphx.argument(val))

    # run the model
    model_outputs = model.run(pp)

    outputs = []
    for output in model_outputs:
        host_output = migraphx.from_gpu(output)
        outputs.append(np.array(host_output))
    return outputs


def check_correctness(gold_outputs, outputs, rtol=1e-3, atol=1e-3):
    if len(gold_outputs) != len(outputs):
        print("Number of outputs {} is not equal to expected number {}".format(
            len(outputs), len(gold_outputs)))
        return False

    out_num = len(gold_outputs)
    ret = True
    for i in range(out_num):
        if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
            print("\nOutput {} is incorrect ...".format(i))
            print("Expected value: \n{}".format(gold_outputs[i]))
            print("......")
            print("Actual value: \n{}\n".format(outputs[i]))
            ret = False

    return ret


# check if model input parameter shapes same as data
# if not the same put it in the return dict
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
    target = args.target
    default_dd_val = parse_dyn_dim_str(args.default_dyn_dim_value)
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

    # read and compile model
    if (args.map_dyn_input_dims is not None):
        map_val = parse_map_dyn_input_str(args.map_dyn_input_dims)
        model = migraphx.parse_onnx(model_path_name,
                                    default_dyn_dim_value=default_dd_val,
                                    map_dyn_input_dims=map_val)
    else:
        model = migraphx.parse_onnx(model_path_name,
                                    default_dyn_dim_value=default_dd_val)
    model.compile(migraphx.get_target(target), offload_copy=False)

    # get test cases
    cases = get_test_cases(test_loc)
    case_num = len(cases)
    correct_num = 0
    for case_name in cases:
        io_folder = test_loc + '/' + case_name
        input_data = wrapup_inputs(
            io_folder, param_names)  # {parameter: data as np.array}
        gold_outputs = read_outputs(
            io_folder, output_names)  # [data in output name order]

        # run the model and return outputs
        output_data = run_one_case(model, input_data)

        # check output correctness
        ret = check_correctness(gold_outputs, output_data)
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

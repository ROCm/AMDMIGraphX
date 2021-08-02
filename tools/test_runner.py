import os
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

    return np_array


def wrapup_inputs(io_folder, parameter_names):
    index = 0
    param_map = {}
    for param_name in parameter_names:
        file_name = io_folder + '/input_' + str(index) + '.pb'
        data = read_pb_file(file_name)
        param_map[param_name] = data
        index = index + 1

    return param_map


def read_outputs(io_folder, out_num):
    outputs = []
    for i in range(out_num):
        file_name = io_folder + '/output_' + str(i) + '.pb'
        data = read_pb_file(file_name)
        outputs.append(data)

    return outputs


def run_one_case(model, param_map):
    # convert np array to model argument
    pp = {}
    for key, val in param_map.items():
        print("input = {}".format(val))
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
        print("Expected value: \n{}".format(gold_outputs[i]))
        print("Actual value: \n{}".format(outputs[i]))
        if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
            print("Output {} is incorrect ...".format(i))
            print("Expected value: \n{}".format(gold_outputs[i]))
            print("Actual value: \n{}".format(outputs[i]))
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
    # read and compile model
    model = migraphx.parse_onnx(model_path_name)
    param_names = model.get_parameter_names()
    output_shapes = model.get_output_shapes()

    model.compile(migraphx.get_target(target))

    # get test cases
    cases = get_test_cases(test_loc)
    case_num = len(cases)
    correct_num = 0
    for case_name in cases:
        io_folder = test_loc + '/' + case_name
        input_data = wrapup_inputs(io_folder, param_names)
        gold_output_data = read_outputs(io_folder, len(output_shapes))

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
        ret = check_correctness(gold_output_data, output_data)
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

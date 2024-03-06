import numpy as np
import onnx
import argparse
import sys
from onnx import helper
from onnx import TensorProto
from onnx.numpy_helper import from_array
from onnx.checker import check_model
import os
from pandas import DataFrame
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=int, nargs="+")
    parser.add_argument('--filter', type=int, nargs="+")
    parser.add_argument('--padding', type=int, nargs="+")
    parser.add_argument('--stride', type=int, nargs="+")
    parser.add_argument('--dilation', type=int, nargs="+")
    parser.add_argument('--group', type=int)
    parser.add_argument('--padding_mode', type=int)
    return parser


def gen_graph(op_info, name):
    graph_def = helper.make_graph(op_info[0], name,
                                  op_info[1], op_info[2])
    model_def = helper.make_model(graph_def,
                                  producer_name=name)
    onnx.shape_inference.infer_shapes(model_def)
    check_model(model_def)
    onnx.save_model(model_def,
                    '{}.onnx'.format(name),
                    save_as_external_data=False,
                    location='{}.weight'.format(name),
                    size_threshold=0,
                    convert_attribute=True)


def gen_conv(activation_lens: list[int], filter_lens: list[int], padding: list[int], stride: list[int],  dilation: list[int], name: str):
    x = helper.make_tensor_value_info(
        '0', TensorProto.FLOAT16, activation_lens)
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT16, filter_lens)
    out = helper.make_tensor_value_info(
        '2', TensorProto.FLOAT16, [None, None, None, None])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 dilations=dilation,
                                 strides=stride,
                                 pads=padding)

    gen_graph(([node], [x, y], [out]), name)


if __name__ == "__main__":
    conv_config_file = open(sys.argv[1], 'r')
    conv_configs = conv_config_file.readlines()
    parser = parse_args()
    onnx_counter = 0
    activation_list = []
    filter_list = []
    padding_list = []
    stride_list = []
    dilation_list = []
    conv_perf_list = []
    total_time_list = []
    rate_list = []
    for config in conv_configs[:2]:
        args = parser.parse_args(config.split())
        activation_list.append(args.activation)
        filter_list.append(args.filter)
        padding_list.append(args.padding)
        stride_list.append(args.stride)
        dilation_list.append(args.dilation)
        name = "conv_unet_"+str(onnx_counter)
        onnx_counter = onnx_counter+1
        gen_conv(args.activation, args.filter, args.padding,
                 args.stride, args.dilation, name)
        command_list = ['/home/umayadav/repo/AMDMIGraphX/build/bin/driver',
                        'perf', name+".onnx"]
        result = subprocess.run(
            args=command_list, capture_output=True).stdout.decode('utf-8')
        summary = result[result.find("Summary:"):].splitlines()
        for i in summary:
            if i.startswith("gpu::convolution:"):
                conv_perf_list.append(i.split(' ')[1])
            elif i.startswith("gpu::code_object::mlir_convolution:"):
                conv_perf_list.append(i.split(' ')[1])
            elif i.startswith("Rate:"):
                rate_list.append(i.split(' ')[1])
            elif i.startswith("Total time:"):
                total_time_list.append(i.split(' ')[2])
    d = {'activation_size': activation_list, 'filter_size': filter_list, 'padding': padding_list,
         'stride': stride_list, 'dilation': dilation_list, 'conv_time': conv_perf_list, 'rate': rate_list, 'total_time': total_time_list}
    df = DataFrame(data=d)
    df.to_excel('miopen_unet_conv_summary.xlsx', sheet_name='miopen_conv_nchw')

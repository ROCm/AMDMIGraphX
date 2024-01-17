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
# Inference with ONNX Runtime
import onnxruntime
import time
from torchvision import models, transforms as T
import torch
from PIL import Image
import numpy as np
import argparse
import os
import subprocess

#Use most upto date weights
inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, progress=True)

# Download ImageNet labels
#!curl -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

def parse_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp16",
        action="store_true",
        required=False,
        default=False,
        help='Perform fp16 quantization on the model before running inference',
    )

    parser.add_argument(
        "--image_dir",
        required=False,
        default="../dataset/images",
        help='Target DIR for images to infer. Default is ../dataset/images'
    )

    parser.add_argument(
        "--batch",
        required=False,
        default=1,
        help='Batch size of images per inference',
        type=int
    )

    parser.add_argument(
        "--top",
        required=False,
        default=1,
        help='Show top K of inference results',
        type=int
    )

    parser.add_argument(
        "--QPS",
        action="store_true",
        required=False,
        default=False,
        help='Show inference result in Queries-Per-Second QPS instead of inference duration (milliseconds)',
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help='Show verbose output',
    )
    
    return parser.parse_args()

def get_image_list_in_dir(dir):
    proc = subprocess.run("ls",
                          shell=True,
                          stdout=subprocess.PIPE,
                          cwd=dir)
    fileList = proc.stdout.decode().split('\n')
    fileList
    return fileList

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, categories, latency, inputs, topk, batch_size):
    
    io_binding = session.io_binding()
    io_binding.bind_cpu_input('input', inputs.cpu().detach().numpy())
    io_binding.bind_output('output')
    start = time.time()
    session.run_with_iobinding(io_binding)
    latency.append(time.time() - start)
    ort_outputs = io_binding.copy_outputs_to_cpu()[0]

    #Get prediction for each item
    for i in range(batch_size):
        output = ort_outputs[i].copy().flatten()
        output = softmax(output)  # this is optional
        top5_catid = np.argsort(-output)[:topk]

        for catid in top5_catid:
            print(categories[catid], output[catid])

def main():
    flags = parse_input_args()

    if flags.verbose:
        print(flags)

    if flags.verbose:
        print("Reading in Imagenet classes")

    # Read the categories
    with open("../dataset/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    if flags.verbose:
        print("Getting Exported Model from Torch")

    # Export the model to ONNX
    image_height = 299
    image_width = 299
    x = torch.randn(flags.batch, 3, image_height, image_width, requires_grad=True)
    #torch_out = inception_v3(x)
    torch.onnx.export(
        inception_v3,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "inception_v3.onnx",  # where to save the model (can be a file or file-like object)
        export_params=
        True,  # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=
        True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'])  # the model's output names

    # Quantize the model
    if flags.fp16:
        if flags.verbose:
            print("FP16 Quantization Enabled")
        os.environ["ORT_MIGRAPHX_FP16_ENABLE"] = "1"  # Enable FP16 precision
    else:
        os.environ["ORT_MIGRAPHX_FP16_ENABLE"] = "0"  # Disable FP32 precision

    session_ops = onnxruntime.SessionOptions()
    if flags.verbose:
        session_ops.log_verbosity_level = 0
        session_ops.log_severity_level = 0

    session_fp32 = onnxruntime.InferenceSession(
        "inception_v3.onnx", providers=['MIGraphXExecutionProvider'], sess_options=session_ops)

    if flags.verbose:
        print("Preprocessing Batched Images")


    # Preproccess and batch images
    latency = []

    if flags.verbose:
        print("Read from dir " + flags.image_dir)
        
    fileList = get_image_list_in_dir(flags.image_dir)
    
    if flags.verbose:
        print(fileList)    


    #Setup input data feed
    input_batch = torch.empty(flags.batch, 3,image_width,image_height)

    batch_size = 0
    for img in fileList:
        filename = img  # change to your filename

        if img == '':
            break

        if flags.verbose:
            print("Preprocess: " + img)

        input_image = Image.open(str(flags.image_dir + "/" + img))
        preprocess = T.Compose([
            T.Resize(342),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch[batch_size] = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        batch_size = batch_size + 1
        if batch_size >= flags.batch:
            break

    if flags.verbose:
        print("Running samples")
    output = run_sample(session_fp32, categories, latency, input_batch, flags.top, batch_size)

    if flags.verbose:
        print("Running Complete")
        print(latency)

    if flags.QPS:
        print("inception_v3, Rate = {} QPS".format(
            format( ( ((flags.batch)) / (sum(latency) / len(latency))) , '.2f')))
    else:    
        print("inception_v3, time = {} ms".format(
            format(sum(latency) * 1000 / len(latency), '.2f')))


if __name__ == "__main__":
    main()


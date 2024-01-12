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
# Inference with ONNX Runtime
import onnxruntime
import time
from torchvision import models, transforms as T
import torch
from PIL import Image
import numpy as np
import argparse
import os

resnet50 = models.resnet50(pretrained=True)

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
        "--image-dir",
        required=False,
        default="",
        help='Target DIR for images to infer. Default is ./images'
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
        default=True,
        help='Show inference result in Queries-Per-Second QPS instead of inference duration (milliseconds)',
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        default=1,
        help='Show verbose output',
    )
    
    return parser.parse_args()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, image_file, categories, latency, inputs, topk):
    start = time.time()
    input_arr = inputs.cpu().detach().numpy()
    ort_outputs = session.run([], {'input': input_arr})[0]
    latency.append(time.time() - start)
    output = ort_outputs.flatten()
    output = softmax(output)  # this is optional
    top5_catid = np.argsort(-output)[:topk]

    for catid in top5_catid:
        print(categories[catid], output[catid])
    return ort_outputs

def main():
    flags = parse_input_args()

    if flags.verbose:
        print(flags)

    if flags.verbose:
        print("Reading in Imagenet classes")

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    if flags.verbose:
        print("Getting Exported Model from Torch")

    # Export the model to ONNX
    image_height = 224
    image_width = 224
    x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
    torch_out = resnet50(x)
    torch.onnx.export(
        resnet50,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "resnet50.onnx",  # where to save the model (can be a file or file-like object)
        export_params=
        True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=
        True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'])  # the model's output names

    # Pre-processing for ResNet-50 Inferencing, from https://pytorch.org/hub/pytorch_vision_resnet/
    resnet50.eval()


    # Quantize the model
    if flags.fp16:
        if flags.verbose:
            print("FP16 Quantization Enabled")
        os.environ["ORT_MIGRAPHX_FP16_ENABLE"] = "1"  # Enable FP16 precision
    else:
        os.environ["ORT_MIGRAPHX_FP16_ENABLE"] = "0"  # Disable FP32 precision

    session_fp32 = onnxruntime.InferenceSession(
        "resnet50.onnx", providers=['MIGraphXExecutionProvider'])

    if flags.verbose:
        print("Preprocessing Batched Images")


    # Preproccess and batch images
    latency = []

    filename = 'images/guitar3.jpg'  # change to your filename

    input_image = Image.open(filename)
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0)  # create a mini-batch as expected by the model


    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        if flags.verbose:
            print("Pushing Batched data and model to GPU")

        input_batch = input_batch.to('cuda')
        resnet50.to('cuda')


    run_sample(session_fp32, 'guitar3.jpg', categories, latency, input_batch, flags.top)

    if flags.QPS:
        print("resnet50, QPS = {} ".format(
            format( 3600*(sum(latency) * 1000 / len(latency)) / flags.batch, '.2f')))
    else:    
        print("resnet50, time = {} ms".format(
            format(sum(latency) * 1000 / len(latency), '.2f')))


if __name__ == "__main__":
    main()



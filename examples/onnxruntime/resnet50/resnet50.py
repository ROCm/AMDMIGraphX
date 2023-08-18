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

resnet50 = models.resnet50(pretrained=True)

# Download ImageNet labels
#!curl -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

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
filename = 'guitar3.jpg'  # change to your filename

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
print("GPU Availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    resnet50.to('cuda')

session_fp32 = onnxruntime.InferenceSession(
    "resnet50.onnx", providers=['MIGraphXExecutionProvider'])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


latency = []


def run_sample(session, image_file, categories, inputs):
    start = time.time()
    input_arr = inputs.cpu().detach().numpy()
    ort_outputs = session.run([], {'input': input_arr})[0]
    latency.append(time.time() - start)
    output = ort_outputs.flatten()
    output = softmax(output)  # this is optional
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])
    return ort_outputs


ort_output = run_sample(session_fp32, 'guitar3.jpg', categories, input_batch)
print("resnet50, time = {} ms".format(
    format(sum(latency) * 1000 / len(latency), '.2f')))

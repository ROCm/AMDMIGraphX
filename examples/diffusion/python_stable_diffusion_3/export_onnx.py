#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the 'Software'), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from argparse import ArgumentParser
import torch
from diffusers import StableDiffusion3Pipeline
import os


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="models/sd3",
        help=
        "Path to save the onnx model. Use it to override the default models/sd3 path."
    )
    return parser.parse_args()


def export_encoders(output_path):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16)
    x = torch.randint(1, (1, 77))
    encoder_path = output_path + '/text_encoder/model.onnx'
    encoder_2_path = output_path + '/text_encoder_2/model.onnx'
    encoder_3_path = output_path + '/text_encoder_3/model.onnx'
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    os.makedirs(os.path.dirname(encoder_2_path), exist_ok=True)
    os.makedirs(os.path.dirname(encoder_3_path), exist_ok=True)

    torch.onnx.export(pipe.text_encoder,
                      x,
                      encoder_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      dynamic_axes={'input_ids': {
                          0: 'batch_size'
                      }})
    torch.onnx.export(pipe.text_encoder_2,
                      x,
                      encoder_2_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      dynamic_axes={'input_ids': {
                          0: 'batch_size'
                      }})
    torch.onnx.export(pipe.text_encoder_3,
                      x,
                      encoder_3_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      dynamic_axes={'input_ids': {
                          0: 'batch_size'
                      }})


if __name__ == "__main__":
    args = argparser()
    export_encoders(**vars(args))

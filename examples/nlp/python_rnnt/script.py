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

# Model Download Instructions from diffusers/scripts/
# python3 convert_stable_diffusion_controlnet_to_onnx.py \
#     --model_path runwayml/stable-diffusion-v1-5 \
#     --controlnet_path lllyasviel/sd-controlnet-canny \
#     --output_path sd15-onnx \
#     --fp16

from argparse import ArgumentParser
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms

import sys
mgx_lib_path = "/opt/rocm/lib/"
# or if you locally built MIGraphX
mgx_lib_path = "/code/AMDMIGraphX/build/lib/"
if mgx_lib_path not in sys.path:
    sys.path.append(mgx_lib_path)
import migraphx as mgx

import migraphx as mgx
import os
import sys
import torch
import time
from functools import wraps

from hip import hip
from collections import namedtuple
HipEventPair = namedtuple('HipEventPair', ['start', 'end'])

# measurement helper
def measure(fn):
    @wraps(fn)
    def measure_ms(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter_ns()
        print(
            f"Elapsed time for {fn.__name__}: {(end_time - start_time) * 1e-6:.4f} ms\n"
        )
        return result

    return measure_ms

mgx_to_torch_dtype_dict = {
    "bool_type": torch.bool,
    "uint8_type": torch.uint8,
    "int8_type": torch.int8,
    "int16_type": torch.int16,
    "int32_type": torch.int32,
    "int64_type": torch.int64,
    "float_type": torch.float32,
    "double_type": torch.float64,
    "half_type": torch.float16,
}

torch_to_mgx_dtype_dict = {
    value: key
    for (key, value) in mgx_to_torch_dtype_dict.items()
}


def tensor_to_arg(tensor):
    return mgx.argument_from_pointer(
        mgx.shape(
            **{
                "type": torch_to_mgx_dtype_dict[tensor.dtype],
                "lens": list(tensor.size()),
                "strides": list(tensor.stride())
            }), tensor.data_ptr())


def tensors_to_args(tensors):
    return {name: tensor_to_arg(tensor) for name, tensor in tensors.items()}


def get_output_name(idx):
    return f"main:#output_{idx}"


def copy_tensor_sync(tensor, data):
    tensor.copy_(data)
    torch.cuda.synchronize()


def run_model_sync(model, args):
    model.run(args)
    mgx.gpu_sync()


def allocate_torch_tensors(model):
    input_shapes = model.get_parameter_shapes()
    data_mapping = {
        name: torch.zeros(shape.lens()).to(
            mgx_to_torch_dtype_dict[shape.type_string()]).to(device="cuda")
        for name, shape in input_shapes.items()
    }
    return data_mapping


class RNNT_MGX():
    def __init__(self):

        fp16 = ["rnnt_encoder", "rnnt_prediction", "rnnt_joint"]

        compiled_model_path = ''
        onnx_model_path = 'models/rnnt/'
        force_compile = False
        exhaustive_tune = False
        print("Load models...")

        self.models = {
            "rnnt_encoder":
            RNNT_MGX.load_mgx_model(
                "rnnt_encoder", {
                    "input": [157, 1, 240], # seq_length, batch_size, feature_length
                    "feature_length": [157]
                },
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="rnnt_encoder" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "rnnt_prediction":
            RNNT_MGX.load_mgx_model(
                "rnnt_prediction", {
                    "symbol": [1, 20],
                    "hidden_in_1": [2, 1, 320],
                    "hidden_in_2": [2, 1, 320]
                },
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="rnnt_prediction" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "rnnt_joint":
            RNNT_MGX.load_mgx_model(
                "rnnt_joint", {
                    "0": [1, 1, 1024],
                    "1": [1, 320]
                },
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="rnnt_joint" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
        }

        self.tensors = {
            "rnnt_encoder": allocate_torch_tensors(self.models["rnnt_encoder"]),
            "rnnt_prediction": allocate_torch_tensors(self.models["rnnt_prediction"]),
            "rnnt_joint": allocate_torch_tensors(self.models["rnnt_joint"]),
        }

        self.model_args = {
            "rnnt_encoder": tensors_to_args(self.tensors['rnnt_encoder']),
            "rnnt_prediction": tensors_to_args(self.tensors['rnnt_prediction']),
            "rnnt_joint": tensors_to_args(self.tensors['rnnt_joint']),
        }


    @measure
    @torch.no_grad()
    def run(self, seed=13):

        seq_length, batch_size, feature_length = 157, 1, 240

        inp = torch.randn(
            (seq_length, batch_size, feature_length),
            generator=torch.manual_seed(seed)).to(device="cuda")
        feature_length = torch.LongTensor([seq_length]).to(device="cuda")

        copy_tensor_sync(self.tensors["rnnt_encoder"]["input"],
                            inp.to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_encoder"]["feature_length"],
                            feature_length.to(torch.int32))
        run_model_sync(self.models["rnnt_encoder"], self.model_args["rnnt_encoder"])
        
        x_padded, x_lens = self.tensors["rnnt_encoder"][get_output_name(0)], self.tensors["rnnt_encoder"][get_output_name(1)]

        symbol = torch.LongTensor([[20]]).to(device="cuda")
        hidden = torch.randn([2, batch_size, 320]).to(device="cuda"), torch.randn([2, batch_size, 320]).to(device="cuda")

        copy_tensor_sync(self.tensors["rnnt_prediction"]["symbol"],
                            symbol.to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["hidden_in_1"],
                            hidden[0].to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["hidden_in_2"],
                            hidden[1].to(torch.int32))
        run_model_sync(self.models["rnnt_prediction"], self.model_args["rnnt_prediction"])

        g, hidden = self.tensors["rnnt_prediction"][get_output_name(0)], self.tensors["rnnt_prediction"][get_output_name(1)]

        f = torch.randn([batch_size, 1, 1024]).to(device="cuda")

        copy_tensor_sync(self.tensors["rnnt_joint"]["onnx::Shape_0"],
                            f.to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_joint"]["onnx::Shape_1"],
                            g.to(torch.int32))
        
        run_model_sync(self.models["rnnt_joint"], self.model_args["rnnt_joint"])
        result = self.tensors["rnnt_joint"][get_output_name(0)]

  
    @staticmethod
    @measure
    def load_mgx_model(name,
                       shapes,
                       onnx_model_path,
                       compiled_model_path=None,
                       use_fp16=False,
                       force_compile=False,
                       exhaustive_tune=False,
                       offload_copy=True):
        print(f"Loading {name} model...")
        if compiled_model_path is None:
            compiled_model_path = onnx_model_path
        onnx_file = f"{onnx_model_path}/{name}/model.onnx"
        mxr_file = f"{compiled_model_path}/{name}/model_{'fp16' if use_fp16 else 'fp32'}_{'gpu' if not offload_copy else 'oc'}.mxr"
        if not force_compile and os.path.isfile(mxr_file):
            print(f"Found mxr, loading it from {mxr_file}")
            model = mgx.load(mxr_file, format="msgpack")
        elif os.path.isfile(onnx_file):
            print(f"No mxr found at {mxr_file}")
            print(f"Parsing from {onnx_file}")
            model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
            if use_fp16:
                mgx.quantize_fp16(model)
            model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=exhaustive_tune,
                          offload_copy=offload_copy)
            print(f"Saving {name} model to {mxr_file}")
            os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
            mgx.save(model, mxr_file, format="msgpack")
        else:
            print(
                f"No {name} model found at {onnx_file} or {mxr_file}. Please download it and re-try."
            )
            sys.exit(1)
        return model

    @measure
    def warmup(self, num_runs):
        copy_tensor_sync(self.tensors["rnnt_encoder"]["input"],
                         torch.ones((157, 1, 240)).to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_encoder"]["feature_length"],
                         torch.randn((157)).to(torch.float32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["symbol"],
                         torch.randn((1, 20)).to(torch.float32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["hidden_in_1"],
                         torch.randn((2, 1, 320)).to(torch.float32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["hidden_in_2"],
                         torch.randn((2, 1, 320)).to(torch.float32))
        copy_tensor_sync(self.tensors["rnnt_joint"]["onnx::Shape_0"],
                         torch.randn((1, 1, 1024)).to(torch.float32))
        copy_tensor_sync(self.tensors["rnnt_joint"]["onnx::Shape_1"],
                         torch.randn((1, 320)).to(torch.float32))
        
        for _ in range(num_runs):
            run_model_sync(self.models["rnnt_encoder"], self.model_args["rnnt_encoder"])
            run_model_sync(self.models["rnnt_prediction"], self.model_args["rnnt_prediction"])
            run_model_sync(self.models["rnnt_joint"],
                           self.model_args["rnnt_joint"])



if __name__ == "__main__":
    sd = RNNT_MGX()
    print("Warmup")
    sd.warmup(5)
    print("Run")
    sd.run()
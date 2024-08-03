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
from typing import List, Optional, Tuple

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
import torch.nn.functional as F
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

        fp16 = [] #"rnnt_encoder", "rnnt_prediction", "rnnt_joint"]

        compiled_model_path = None
        onnx_model_path = 'models/rnnt/'
        force_compile = False
        exhaustive_tune = False
        print("Load models...")

        self.models = {
            "rnnt_encoder":
            RNNT_MGX.load_mgx_model(
                "rnnt_encoder", {
                    "input": [70, 1, 240], # seq_length, batch_size, feature_length
                    "feature_length": [1]
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
                    "symbol": [1, 1],
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


    # @measure
    @torch.no_grad()
    def encoder(self, inp, feature_length):
        copy_tensor_sync(self.tensors["rnnt_encoder"]["input"],
                            inp.to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_encoder"]["feature_length"],
                            feature_length.to(torch.int32))
        run_model_sync(self.models["rnnt_encoder"], self.model_args["rnnt_encoder"])
        
        x_padded, x_lens = self.tensors["rnnt_encoder"][get_output_name(0)], self.tensors["rnnt_encoder"][get_output_name(1)]
        x_padded = x_padded.squeeze(1)
        x_padded = x_padded.transpose(1, 0)

        return x_padded, x_lens
    

    # @measure
    @torch.no_grad()
    def prediction(self, symbol, hidden):
        copy_tensor_sync(self.tensors["rnnt_prediction"]["symbol"],
                            symbol.to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["hidden_in_1"],
                            hidden[0].to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_prediction"]["hidden_in_2"],
                            hidden[1].to(torch.int32))
        run_model_sync(self.models["rnnt_prediction"], self.model_args["rnnt_prediction"])

        g, hidden = self.tensors["rnnt_prediction"][get_output_name(0)], self.tensors["rnnt_prediction"][get_output_name(1)]
        
        return g, hidden 


    # @measure
    @torch.no_grad()
    def joint(self, f, g):
        copy_tensor_sync(self.tensors["rnnt_joint"]["onnx::Shape_0"],
                            f.to(torch.int32))
        copy_tensor_sync(self.tensors["rnnt_joint"]["onnx::Shape_1"],
                            g.to(torch.int32))
        
        run_model_sync(self.models["rnnt_joint"], self.model_args["rnnt_joint"])
        result = self.tensors["rnnt_joint"][get_output_name(0)]
        return result 

  
    @staticmethod
    # @measure
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
    
def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict

class ScriptGreedyDecoder():
    def __init__(self, max_symbols_per_step=5):
        self._alt_model = RNNT_MGX()
        self._SOS = -1
        self._blank_id = 28
        self._max_symbols_per_step = max_symbols_per_step

        mlcommons_inference_path = './inference/'
        sys.path.insert(0, mlcommons_inference_path + 'retired_benchmarks/speech_recognition/rnnt/pytorch')
        from model_separable_rnnt import RNNT
        import toml 

        config_toml = 'inference/retired_benchmarks/speech_recognition/rnnt/pytorch/configs/rnnt.toml'
        config = toml.load(config_toml)

        def add_blank_label(labels):
            if not isinstance(labels, list):
                raise ValueError("labels must be a list of symbols")
            labels.append("<BLANK>")
            return labels

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self._model = RNNT(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab)
        )
        self._model.load_state_dict(load_and_migrate_checkpoint('rnnt.pt'),
                              strict=True)
        self._model.to('cuda')
        self._model.eval()

    def run(self, x: torch.Tensor, out_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        logits, logits_lens = self._alt_model.encoder(x, out_lens)

        output: List[List[int]] = []
        for batch_idx in range(logits.size(0)):
            inseq = logits[batch_idx, :, :].unsqueeze(1).to('cuda')
            # inseq: TxBxF
            logitlen = logits_lens[batch_idx]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)

        return logits, logits_lens, output

    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor) -> List[int]:
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        label: List[int] = []
        for time_idx in range(int(out_len.item())):
            f = x[time_idx, :, :].unsqueeze(0).to('cuda')

            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < self._max_symbols_per_step:
                
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden
                )

                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime

                symbols_added += 1

        print('returning', label)
        return label

    def _pred_step(self, label: int, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if label == self._SOS:
            label = 0
        # if label > self._blank_id:
        #     label -= 1

        label = torch.tensor([[label]], dtype=torch.int64)

        if hidden is None:
            hidden = torch.zeros([2, 1, 320]).to(device="cuda"), torch.zeros([2, 1, 320]).to(device="cuda")
       
        return self._alt_model.prediction(label, hidden)


    def _joint_step(self, enc: torch.Tensor, pred: torch.Tensor, log_normalize: bool=False) -> torch.Tensor:
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
   
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels: List[int]) -> int:
        return self._SOS if len(labels) == 0 else labels[-1]


if __name__ == "__main__":
    sd = ScriptGreedyDecoder()

    x = torch.load('feature.pt').to("cuda")
    out_lens = torch.load('feature_length.pt').to("cuda")

    #seq_length, batch_size, feature_length = 157, 1, 240
    #x = torch.randn((seq_length, batch_size, feature_length)).to(device="cuda")
    #out_lens = torch.LongTensor([seq_length]).to(device="cuda")

    result = sd.run(x, out_lens)
    print(result)
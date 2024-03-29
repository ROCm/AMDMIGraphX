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
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from PIL import Image

import migraphx as mgx
import numpy as np
import os
import sys
import torch
import time
from functools import wraps

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


def allocate_torch_buffers(model):
    input_shapes = model.get_parameter_shapes()
    data_mapping = {
        name: torch.zeros(shape.lens()).to(
            mgx_to_torch_dtype_dict[shape.type_string()]).to(device="cuda")
        for name, shape in input_shapes.items()
    }
    return data_mapping


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


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="./sdxl-1.0-base",
        help="Path to onnx model exports",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "-t",
        "--steps",
        type=int,
        default=30,
        help="Number of steps",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        help="Prompt",
    )

    parser.add_argument(
        "-n",
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="Guidance scale",
    )

    parser.add_argument(
        "--save-compiled",
        action="store_true",
        default=False,
        help="Save compiled programs as .mxr files",
    )

    parser.add_argument(
        "--exhaustive-tune",
        action="store_true",
        default=False,
        help="Perform exhaustive tuning when compiling onnx models",
    )

    parser.add_argument(
        "--vae-fp32",
        action="store_true",
        default=False,
        help="Use fp32 version of VAE",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name",
    )
    return parser.parse_args()


class StableDiffusionMGX():
    def __init__(self, base_model_path, save_compiled, vae_fp32, exhaustive_tune=False):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Using {model_id}")

        print("Creating EulerDiscreteScheduler scheduler")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler")

        print("Creating CLIPTokenizer tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id,
                                                       subfolder="tokenizer")

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, use_safetensors=True, subfolder="tokenizer_2")

        print("Load models...")

        self.vae_name = "vae_fp16_fix"
        self.vae_input_name = "input.1"
        if vae_fp32:
            self.vae_name = "vae"
            self.vae_input_name = "latent_sample"
        self.vae = StableDiffusionMGX.load_mgx_model(
            self.vae_name, {self.vae_input_name: [1, 4, 128, 128]},
            base_model_path,
            save_compiled,
            exhaustive_tune,
            offload_copy=False)
        self.clip = StableDiffusionMGX.load_mgx_model("clip",
                                                      {"input_ids": [1, 77]},
                                                      base_model_path,
                                                      save_compiled,
                                                      exhaustive_tune,
                                                      offload_copy=False)
        self.clip2 = StableDiffusionMGX.load_mgx_model("clip2",
                                                       {"input_ids": [1, 77]},
                                                       base_model_path,
                                                       save_compiled,
                                                       exhaustive_tune,
                                                       offload_copy=False)
        self.unetxl = StableDiffusionMGX.load_mgx_model("unetxl.opt", {
            "sample": [2, 4, 128, 128],
            "encoder_hidden_states": [2, 77, 2048],
            "text_embeds": [2, 1280],
            "time_ids": [2, 6],
            "timestep": [1],
        },
                                                        base_model_path,
                                                        save_compiled,
                                                        exhaustive_tune,
                                                        offload_copy=False)

        self.tensors = {
            "clip": allocate_torch_buffers(self.clip),
            "clip2": allocate_torch_buffers(self.clip2),
            "unetxl": allocate_torch_buffers(self.unetxl),
            "vae": allocate_torch_buffers(self.vae),
        }

        self.model_args = {
            "clip": tensors_to_args(self.tensors['clip']),
            "clip2": tensors_to_args(self.tensors['clip2']),
            "unetxl": tensors_to_args(self.tensors['unetxl']),
            "vae": tensors_to_args(self.tensors['vae']),
        }

    @measure
    def run(self, prompt, negative_prompt, steps, seed, scale):
        with torch.inference_mode():
            # need to set this for each run
            self.scheduler.set_timesteps(steps, device="cuda")

            print("Tokenizing prompt...")
            text_input = self.tokenize(prompt, False)
            text_input2 = self.tokenize(prompt, True)

            print("Tokenizing negative prompt...")
            uncond_input = self.tokenize(negative_prompt, False)
            uncond_input2 = self.tokenize(negative_prompt, True)

            start_time = time.perf_counter_ns()
            print("Creating text embeddings for prompt...")
            text_embeddings = self.get_embeddings(text_input, text_input2)

            print("Creating text embeddings for negative prompt...")
            uncond_embeddings = self.get_embeddings(uncond_input,
                                                    uncond_input2)
            end_time = time.perf_counter_ns()
            clip_time = end_time - start_time

            print(
                f"Creating random input data ({1}x{4}x{128}x{128}) (latents) with seed={seed}..."
            )
            latents = torch.randn(
                (1, 4, 128, 128),
                generator=torch.manual_seed(seed)).to(device="cuda")

            print("Apply initial noise sigma\n")
            latents = latents * self.scheduler.init_noise_sigma

            start_time = time.perf_counter_ns()
            print("Running denoising loop...")
            time_id = torch.tensor([[1024, 1024, 0, 0, 1024, 1024],
                                    [1024, 1024, 0, 0, 1024, 1024]
                                    ]).to(torch.float16).to(device="cuda")
            # time_id = torch.ones((2, 6)).to(torch.float16)

            hidden_states = torch.concatenate(
                (text_embeddings[0], uncond_embeddings[0])).to(torch.float16)
            text_embeds = torch.concatenate(
                (text_embeddings[1], uncond_embeddings[1])).to(torch.float16)

            for step, t in enumerate(self.scheduler.timesteps):
                print(f"#{step}/{len(self.scheduler.timesteps)} step")
                latents = self.denoise_step(text_embeds, hidden_states,
                                            latents, t, scale, time_id)
            end_time = time.perf_counter_ns()
            unet_time = end_time - start_time

            print("Scale denoised result...")
            latents = 1 / 0.18215 * latents

            print("Decode denoised result...")
            start_time = time.perf_counter_ns()
            image = self.decode(latents)
            end_time = time.perf_counter_ns()
            vae_time = end_time - start_time

            print(f"Elapsed time clip: {(clip_time) * 1e-6:.4f} ms\n")
            print(f"Elapsed time unet: {(unet_time) * 1e-6:.4f} ms\n")
            print(f"Elapsed time vae: {(vae_time) * 1e-6:.4f} ms\n")

            image = image.detach().clone().cpu().numpy()
            return image

    @staticmethod
    @measure
    def load_mgx_model(name,
                       shapes,
                       model_base_path,
                       save_compiled,
                       exhaustive_tune=False,
                       offload_copy=True):
        file = f"{model_base_path}/{name}/model"
        mxr_file = f"{file}{'_gpu' if not offload_copy else '_oc'}.mxr"
        print(f"Loading {name} model from {file}")
        if os.path.isfile(mxr_file):
            print("Found mxr, loading it...")
            model = mgx.load(mxr_file, format="msgpack")
        elif os.path.isfile(f"{file}.onnx"):
            print("Parsing from onnx file...")
            model = mgx.parse_onnx(f"{file}.onnx", map_input_dims=shapes)
            if name != "vae":
                mgx.quantize_fp16(model)
            model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=exhaustive_tune,
                          offload_copy=offload_copy)
            if save_compiled:
                print(f"Saving {name} model to mxr file...")
                mgx.save(model, mxr_file, format="msgpack")
        else:
            print(f"No {name} model found. Please download it and re-try.")
            sys.exit(1)
        return model

    @measure
    def tokenize(self, input, is_tokenizer2):
        if is_tokenizer2:
            return self.tokenizer_2([input],
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")
        return self.tokenizer([input],
                              padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt")

    @measure
    def get_embeddings(self, input, input2):
        self.tensors['clip']['input_ids'].copy_(input.input_ids.to(
            torch.int32))
        self.clip.run(self.model_args['clip'])

        self.tensors['clip2']['input_ids'].copy_(
            input2.input_ids.to(torch.int32))
        self.clip2.run(self.model_args['clip2'])
        mgx.gpu_sync()

        clip_hidden = self.tensors['clip'][get_output_name(0)]
        clip2_hidden = self.tensors['clip2'][get_output_name(1)]
        clip2_embed = self.tensors['clip2'][get_output_name(0)]
        return (torch.concatenate((clip_hidden, clip2_hidden),
                                  axis=2), clip2_embed)

    @staticmethod
    def convert_to_rgb_image(image):
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        images = (image * 255).round().astype("uint8")
        return Image.fromarray(images[0])

    @staticmethod
    def save_image(pil_image, filename="output.png"):
        pil_image.save(filename)

    @measure
    def denoise_step_pre(self, latents, t):
        sample = self.scheduler.scale_model_input(latents, t).to(
            torch.float32).to(device="cuda")
        sample = torch.concatenate((sample, sample))
        timestep = torch.atleast_1d(t.to(torch.float32)).to(
            device="cuda")  # convert 0D -> 1D

        return sample, timestep

    @measure
    def denoise_step_infer(self, sample, timestep, hidden_states, text_embeds,
                           time_id):
        self.tensors['unetxl']['sample'].copy_(sample.to(torch.float32))
        self.tensors['unetxl']['encoder_hidden_states'].copy_(
            hidden_states.to(torch.float16))
        self.tensors['unetxl']['timestep'].copy_(timestep.to(torch.float32))
        self.tensors['unetxl']['text_embeds'].copy_(
            text_embeds.to(torch.float16))
        self.tensors['unetxl']['time_ids'].copy_(time_id.to(torch.float16))
        self.unetxl.run(self.model_args['unetxl'])
        mgx.gpu_sync()
        return torch.tensor_split(self.tensors['unetxl'][get_output_name(0)],
                                  2)

    @measure
    def denoise_step_post(self, noise_pred_text, noise_pred_uncond, latents, t,
                          scale):
        # perform guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                  noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(noise_pred, t, latents).prev_sample

    def denoise_step(self, text_embeds, hidden_states, latents, t, scale,
                     time_id):
        sample, timestep = self.denoise_step_pre(latents, t)
        noise_pred_text, noise_pred_uncond = self.denoise_step_infer(
            sample, timestep, hidden_states, text_embeds, time_id)
        return self.denoise_step_post(noise_pred_text, noise_pred_uncond,
                                      latents, t, scale)

    @measure
    def decode(self, latents):
        self.tensors['vae'][self.vae_input_name].copy_(latents.to(torch.float32))
        self.vae.run(self.model_args['vae'])
        mgx.gpu_sync()
        return self.tensors['vae'][get_output_name(0)]

    def warmup(self, num_runs):
        self.tensors['clip']['input_ids'].copy_(
            torch.ones((1, 77)).to(torch.int32))
        self.tensors['clip2']['input_ids'].copy_(
            torch.ones((1, 77)).to(torch.int64))
        self.tensors['unetxl']['sample'].copy_(
            torch.randn((2, 4, 128, 128)).to(torch.float32))
        self.tensors['unetxl']['encoder_hidden_states'].copy_(
            torch.randn((2, 77, 2048)).to(torch.float16))
        self.tensors['unetxl']['timestep'].copy_(
            torch.randn((1)).to(torch.float32))
        self.tensors['unetxl']['text_embeds'].copy_(
            torch.randn((2, 1280)).to(torch.float16))
        self.tensors['unetxl']['time_ids'].copy_(
            torch.randn((2, 6)).to(torch.float16))
        self.tensors['vae'][self.vae_input_name].copy_(
            torch.randn((1, 4, 128, 128)).to(torch.float32))
        for _ in range(num_runs):
            self.clip.run(self.model_args['clip'])
            self.clip2.run(self.model_args['clip2'])
            self.unetxl.run(self.model_args['unetxl'])
            self.vae.run(self.model_args['vae'])
            mgx.gpu_sync()


if __name__ == "__main__":
    args = get_args()

    sd = StableDiffusionMGX(args.base_model_path, args.save_compiled, args.vae_fp32,
                            args.exhaustive_tune)
    print("Warming up...")
    sd.warmup(5)
    print(f"Running inference")
    result = sd.run(args.prompt, args.negative_prompt, args.steps, args.seed,
                    args.scale)

    print("Convert result to rgb image...")
    image = StableDiffusionMGX.convert_to_rgb_image(result)
    filename = args.output if args.output else f"output_s{args.seed}_t{args.steps}_torch.png"
    StableDiffusionMGX.save_image(image, filename)
    print(f"Image saved to {filename}")

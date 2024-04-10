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
import os
import sys
import torch
import time
from functools import wraps


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
    # Model compile
    parser.add_argument(
        "--onnx-model-path",
        type=str,
        default="models/sdxl-1.0-base/",
        help="Path to onnx model files.",
    )

    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default=None,
        help=
        "Path to compiled mxr model files. If not set, it will be saved next to the onnx model.",
    )

    parser.add_argument(
        "--refiner-onnx-model-path",
        type=str,
        default=None,
        help="Path to onnx model files.",
    )

    parser.add_argument(
        "--refiner-compiled-model-path",
        type=str,
        default=None,
        help=
        "Path to compiled mxr model files. If not set, it will be saved next to the refiner onnx model.",
    )

    parser.add_argument(
        "--fp16",
        choices=["all", "vae", "clip", "clip2", "unetxl"],
        nargs="+",
        help="Quantize models with fp16 precision.",
    )

    parser.add_argument(
        "--force-compile",
        action="store_true",
        default=False,
        help="Ignore existing .mxr files and override them",
    )

    parser.add_argument(
        "--exhaustive-tune",
        action="store_true",
        default=False,
        help="Perform exhaustive tuning when compiling onnx models",
    )

    # Runtime
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
        required=True,
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
        "--refiner-aesthetic-score",
        type=float,
        default=6.0,
        help="aesthetic score for refiner",
    )

    parser.add_argument(
        "--refiner-negative-aesthetic-score",
        type=float,
        default=2.5,
        help="negative aesthetic score for refiner",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name",
    )
    return parser.parse_args()


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


class StableDiffusionMGX():
    def __init__(self, onnx_model_path, compiled_model_path,
                 refiner_onnx_model_path, refiner_compiled_model_path, fp16,
                 force_compile, exhaustive_tune):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Using {model_id}")

        print("Creating EulerDiscreteScheduler scheduler")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler")

        print("Creating CLIPTokenizer tokenizers...")
        self.tokenizers = {
            "clip":
            CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
            "clip2":
            CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        }

        if fp16 is None:
            fp16 = []
        elif "all" in fp16:
            fp16 = ["vae", "clip", "clip2", "unetxl"]

        print("Load models...")
        self.models = {
            "vae":
            StableDiffusionMGX.load_mgx_model(
                "vae_decoder", {"latent_sample": [1, 4, 128, 128]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="vae" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "clip":
            StableDiffusionMGX.load_mgx_model(
                "clip.opt.mod", {"input_ids": [2, 77]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="clip" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "clip2":
            StableDiffusionMGX.load_mgx_model(
                "clip2.opt.mod", {"input_ids": [2, 77]},
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="clip2" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False),
            "unetxl":
            StableDiffusionMGX.load_mgx_model(
                "unetxl.opt", {
                    "sample": [2, 4, 128, 128],
                    "encoder_hidden_states": [2, 77, 2048],
                    "text_embeds": [2, 1280],
                    "time_ids": [2, 6],
                    "timestep": [1],
                },
                onnx_model_path,
                compiled_model_path=compiled_model_path,
                use_fp16="unetxl" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)
        }

        self.tensors = {
            "clip": allocate_torch_tensors(self.models["clip"]),
            "clip2": allocate_torch_tensors(self.models["clip2"]),
            "unetxl": allocate_torch_tensors(self.models["unetxl"]),
            "vae": allocate_torch_tensors(self.models["vae"]),
        }

        self.model_args = {
            "clip": tensors_to_args(self.tensors["clip"]),
            "clip2": tensors_to_args(self.tensors["clip2"]),
            "unetxl": tensors_to_args(self.tensors["unetxl"]),
            "vae": tensors_to_args(self.tensors["vae"]),
        }

        self.use_refiner = refiner_onnx_model_path or refiner_compiled_model_path
        if self.use_refiner:
            self.models["refiner_unetxl"] = StableDiffusionMGX.load_mgx_model(
                "unetxl.opt", {
                    "sample": [2, 4, 128, 128],
                    "encoder_hidden_states": [2, 77, 1280],
                    "text_embeds": [2, 1280],
                    "time_ids": [2, 5],
                    "timestep": [1],
                },
                refiner_onnx_model_path,
                compiled_model_path=refiner_compiled_model_path,
                use_fp16="unetxl" in fp16,
                force_compile=force_compile,
                exhaustive_tune=exhaustive_tune,
                offload_copy=False)

            self.tensors["refiner_unetxl"] = allocate_torch_tensors(
                self.models["refiner_unetxl"])
            self.model_args["refiner_unetxl"] = tensors_to_args(
                self.tensors["refiner_unetxl"])

    @measure
    @torch.no_grad()
    def run(self, prompt, negative_prompt, steps, seed, scale,
            refiner_aesthetic_score, refiner_negative_aesthetic_score):
        torch.cuda.synchronize()
        # need to set this for each run
        self.scheduler.set_timesteps(steps, device="cuda")

        print("Tokenizing prompts...")
        prompt_tokens = self.tokenize(prompt, negative_prompt)

        print("Creating text embeddings...")
        hidden_states, text_embeddings = self.get_embeddings(prompt_tokens)
        # The opt version expects fp16 inputs
        hidden_states, text_embeddings = hidden_states.to(
            torch.float16), text_embeddings.to(torch.float16)

        print(
            f"Creating random input data ({1}x{4}x{128}x{128}) (latents) with seed={seed}..."
        )
        noise = torch.randn(
            (1, 4, 128, 128),
            generator=torch.manual_seed(seed)).to(device="cuda")
        # input h/w crop h/w output h/w
        time_id = [1024, 1024, 0, 0, 1024, 1024]
        time_ids = torch.tensor([time_id,
                                 time_id]).to(torch.float16).to(device="cuda")

        print("Apply initial noise sigma\n")
        latents = noise * self.scheduler.init_noise_sigma

        print("Running denoising loop...")
        for step, t in enumerate(self.scheduler.timesteps):
            print(f"#{step}/{len(self.scheduler.timesteps)} step")
            latents = self.denoise_step(text_embeddings,
                                        hidden_states,
                                        latents,
                                        t,
                                        scale,
                                        time_ids,
                                        model="unetxl")

        if self.use_refiner:
            # only use the clip2 part
            hidden_states = hidden_states[:, :, 768:]
            # input h/w crop h/w scores
            time_id_pos = time_id[:4] + [refiner_aesthetic_score]
            time_id_neg = time_id[:4] + [refiner_negative_aesthetic_score]
            time_ids = torch.tensor([time_id_pos, time_id_neg
                                     ]).to(torch.float16).to(device="cuda")
            # need to set this for each run
            self.scheduler.set_timesteps(steps, device="cuda")
            # Add noise to latents using timesteps
            latents = self.scheduler.add_noise(latents, noise,
                                               self.scheduler.timesteps[:1])
            print("Running refiner denoising loop...")
            for step, t in enumerate(self.scheduler.timesteps):
                print(f"#{step}/{len(self.scheduler.timesteps)} step")
                latents = self.denoise_step(text_embeddings,
                                            hidden_states,
                                            latents,
                                            t,
                                            scale,
                                            time_ids,
                                            model="refiner_unetxl")

        print("Scale denoised result...")
        latents = 1 / 0.18215 * latents

        print("Decode denoised result...")
        image = self.decode(latents)

        torch.cuda.synchronize()
        return image

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
    def tokenize(self, prompt, negative_prompt):
        def _tokenize(tokenizer):
            return self.tokenizers[tokenizer](
                [prompt, negative_prompt],
                padding="max_length",
                max_length=self.tokenizers[tokenizer].model_max_length,
                truncation=True,
                return_tensors="pt")

        tokens = _tokenize("clip")
        tokens2 = _tokenize("clip2")
        return (tokens, tokens2)

    @measure
    def get_embeddings(self, prompt_tokens):
        def _create_embedding(model, input):
            copy_tensor_sync(self.tensors[model]["input_ids"],
                             input.input_ids.to(torch.int32))
            run_model_sync(self.models[model], self.model_args[model])

        clip_input, clip2_input = prompt_tokens
        _create_embedding("clip", clip_input)
        _create_embedding("clip2", clip2_input)

        hidden_states = torch.concatenate(
            (self.tensors["clip"][get_output_name(0)],
             self.tensors["clip2"][get_output_name(1)]),
            axis=2)
        text_embeds = self.tensors["clip2"][get_output_name(0)]
        return (hidden_states, text_embeds)

    @staticmethod
    def convert_to_rgb_image(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return Image.fromarray(images[0])

    @staticmethod
    def save_image(pil_image, filename="output.png"):
        pil_image.save(filename)

    @measure
    def denoise_step(self, text_embeddings, hidden_states, latents, t, scale,
                     time_ids, model):
        latents_model_input = torch.cat([latents] * 2)
        latents_model_input = self.scheduler.scale_model_input(
            latents_model_input, t).to(torch.float32).to(device="cuda")
        timestep = torch.atleast_1d(t.to(torch.float32)).to(
            device="cuda")  # convert 0D -> 1D

        copy_tensor_sync(self.tensors[model]["sample"], latents_model_input)
        copy_tensor_sync(self.tensors[model]["encoder_hidden_states"],
                         hidden_states)
        copy_tensor_sync(self.tensors[model]["text_embeds"], text_embeddings)
        copy_tensor_sync(self.tensors[model]["timestep"], timestep)
        copy_tensor_sync(self.tensors[model]["time_ids"], time_ids)
        run_model_sync(self.models[model], self.model_args[model])

        noise_pred_text, noise_pred_uncond = torch.tensor_split(
            self.tensors[model][get_output_name(0)], 2)

        # perform guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                  noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(noise_pred, t, latents).prev_sample

    @measure
    def decode(self, latents):
        copy_tensor_sync(self.tensors["vae"]["latent_sample"], latents)
        run_model_sync(self.models["vae"], self.model_args["vae"])
        return self.tensors["vae"][get_output_name(0)]

    @measure
    def warmup(self, num_runs):

        copy_tensor_sync(self.tensors["clip"]["input_ids"],
                         torch.ones((2, 77)).to(torch.int32))
        copy_tensor_sync(self.tensors["clip2"]["input_ids"],
                         torch.ones((2, 77)).to(torch.int32))
        copy_tensor_sync(self.tensors["unetxl"]["sample"],
                         torch.randn((2, 4, 128, 128)).to(torch.float32))
        copy_tensor_sync(self.tensors["unetxl"]["encoder_hidden_states"],
                         torch.randn((2, 77, 2048)).to(torch.float16))
        copy_tensor_sync(self.tensors["unetxl"]["text_embeds"],
                         torch.randn((2, 1280)).to(torch.float16))
        copy_tensor_sync(self.tensors["unetxl"]["time_ids"],
                         torch.randn((2, 6)).to(torch.float16))
        copy_tensor_sync(self.tensors["unetxl"]["timestep"],
                         torch.randn((1)).to(torch.float32))
        copy_tensor_sync(self.tensors["vae"]["latent_sample"],
                         torch.randn((1, 4, 128, 128)).to(torch.float32))

        for _ in range(num_runs):
            run_model_sync(self.models["clip"], self.model_args["clip"])
            run_model_sync(self.models["clip2"], self.model_args["clip2"])
            run_model_sync(self.models["unetxl"], self.model_args["unetxl"])
            run_model_sync(self.models["vae"], self.model_args["vae"])

        if self.use_refiner:
            copy_tensor_sync(self.tensors["refiner_unetxl"]["sample"],
                             torch.randn((2, 4, 128, 128)).to(torch.float32))
            copy_tensor_sync(
                self.tensors["refiner_unetxl"]["encoder_hidden_states"],
                torch.randn((2, 77, 1280)).to(torch.float16))
            copy_tensor_sync(self.tensors["refiner_unetxl"]["text_embeds"],
                             torch.randn((2, 1280)).to(torch.float16))
            copy_tensor_sync(self.tensors["refiner_unetxl"]["time_ids"],
                             torch.randn((2, 5)).to(torch.float16))
            copy_tensor_sync(self.tensors["refiner_unetxl"]["timestep"],
                             torch.randn((1)).to(torch.float32))
            for _ in range(num_runs):
                run_model_sync(self.models["refiner_unetxl"],
                               self.model_args["refiner_unetxl"])


if __name__ == "__main__":
    args = get_args()

    sd = StableDiffusionMGX(args.onnx_model_path, args.compiled_model_path,
                            args.refiner_onnx_model_path,
                            args.refiner_compiled_model_path, args.fp16,
                            args.force_compile, args.exhaustive_tune)
    sd.warmup(5)
    result = sd.run(args.prompt, args.negative_prompt, args.steps, args.seed,
                    args.scale, args.refiner_aesthetic_score,
                    args.refiner_negative_aesthetic_score)

    print("Convert result to rgb image...")
    image = StableDiffusionMGX.convert_to_rgb_image(result)
    filename = args.output if args.output else f"output_s{args.seed}_t{args.steps}.png"
    StableDiffusionMGX.save_image(image, filename)
    print(f"Image saved to {filename}")

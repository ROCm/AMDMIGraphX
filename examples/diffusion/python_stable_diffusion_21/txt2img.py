#  The MIT License (MIT)
#
#  Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
        default=20,
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
        default=7.0,
        help="Guidance scale",
    )

    parser.add_argument("--fp16",
                        action="store_true",
                        help="Quantize MIGraphX models to fp16")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name",
    )
    return parser.parse_args()


class StableDiffusionMGX():
    def __init__(self, fp16):
        model_id = "stabilityai/stable-diffusion-2-1"
        print(f"Using {model_id}")

        print("Creating EulerDiscreteScheduler scheduler")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler")

        print("Creating CLIPTokenizer tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id,
                                                       subfolder="tokenizer")

        print("Load models...")
        self.vae = StableDiffusionMGX.load_mgx_model(
            "vae_decoder", {"latent_sample": [1, 4, 64, 64]}, fp16)
        self.text_encoder = StableDiffusionMGX.load_mgx_model(
            "text_encoder", {"input_ids": [1, 77]}, fp16)
        self.unet = StableDiffusionMGX.load_mgx_model(
            "unet", {
                "sample": [1, 4, 64, 64],
                "encoder_hidden_states": [1, 77, 1024],
                "timestep": [1],
            }, fp16)

    @measure
    def run(self, prompt, negative_prompt, steps, seed, scale):
        # need to set this for each run
        self.scheduler.set_timesteps(steps)

        print("Tokenizing prompt...")
        text_input = self.tokenize(prompt)

        print("Creating text embeddings for prompt...")
        text_embeddings = self.get_embeddings(text_input)

        print("Tokenizing negative prompt...")
        uncond_input = self.tokenize(negative_prompt)

        print("Creating text embeddings for negative prompt...")
        uncond_embeddings = self.get_embeddings(uncond_input)

        print(
            f"Creating random input data ({1}x{4}x{64}x{64}) (latents) with seed={seed}..."
        )
        latents = torch.randn((1, 4, 64, 64),
                              generator=torch.manual_seed(seed))

        print("Apply initial noise sigma\n")
        latents = latents * self.scheduler.init_noise_sigma

        print("Running denoising loop...")
        latents = self.denoising_loop(text_embeddings, uncond_embeddings,
                                      latents, scale)

        print("Scale denoised result...")
        latents = 1 / 0.18215 * latents

        print("Decode denoised result...")
        image = self.decode(latents)

        return image

    @staticmethod
    @measure
    def load_mgx_model(name, shapes, fp16):
        file = f"models/sd21-onnx/{name}/model"
        if fp16:
            file += "_fp16"
        print(f"Loading {name} model from {file}")
        if os.path.isfile(f"{file}.mxr"):
            print("Found mxr, loading it...")
            model = mgx.load(f"{file}.mxr", format="msgpack")
        elif os.path.isfile(f"{file.rstrip('''_fp16''')}.onnx"):
            print("Parsing from onnx file...")
            model = mgx.parse_onnx(f"{file.rstrip('''_fp16''')}.onnx",
                                   map_input_dims=shapes)
            if fp16:
                mgx.quantize_fp16(model)
            model.compile(mgx.get_target("gpu"))
            print(f"Saving {name} model to mxr file...")
            mgx.save(model, f"{file}.mxr", format="msgpack")
        else:
            print(f"No {name} model found. Please download it and re-try.")
            exit(1)
        return model

    @measure
    def tokenize(self, input):
        return self.tokenizer([input],
                              padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="np")

    @measure
    def get_embeddings(self, input):
        return np.array(
            self.text_encoder.run(
                {"input_ids":
                 input.input_ids.astype(np.int32)})[0]).astype(np.float32)

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
    def denoising_loop(self, text_embeddings, uncond_embeddings, latents,
                       scale):
        for step, t in enumerate(self.scheduler.timesteps):
            print(f"#{step}/{len(self.scheduler.timesteps)} step")
            latents = self.denoise_step(text_embeddings, uncond_embeddings,
                                        latents, t, scale)
        return latents

    @measure
    def denoise_step(self, text_embeddings, uncond_embeddings, latents, t,
                     scale):
        sample = self.scheduler.scale_model_input(latents,
                                                  t).numpy().astype(np.float32)
        timestep = np.atleast_1d(t.numpy().astype(
            np.int64))  # convert 0D -> 1D

        noise_pred_uncond = np.array(
            self.unet.run({
                "sample": sample,
                "encoder_hidden_states": uncond_embeddings,
                "timestep": timestep
            })[0])

        noise_pred_text = np.array(
            self.unet.run({
                "sample": sample,
                "encoder_hidden_states": text_embeddings,
                "timestep": timestep
            })[0])

        # perform guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                  noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(torch.from_numpy(noise_pred), t,
                                   latents).prev_sample

    @measure
    def decode(self, latents):
        return np.array(
            self.vae.run({"latent_sample":
                          latents.numpy().astype(np.float32)})[0])


if __name__ == "__main__":
    args = get_args()

    sd = StableDiffusionMGX(args.fp16)
    result = sd.run(args.prompt, args.negative_prompt, args.steps, args.seed,
                    args.scale)

    print("Convert result to rgb image...")
    image = StableDiffusionMGX.convert_to_rgb_image(result)
    filename = args.output if args.output else f"output_s{args.seed}_t{args.steps}.png"
    StableDiffusionMGX.save_image(image, args.output)
    print(f"Image saved to {filename}")

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
from tqdm.auto import tqdm
from PIL import Image

import migraphx as mgx
import numpy as np
import os
import torch


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

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name",
    )
    return parser.parse_args()


# helper for model loading
def load_mgx_model(name, shapes):
    file = f"models/sd21-onnx/{name}/model"
    print(f"Loading {name} model from {file}")
    if os.path.isfile(f"{file}.mxr"):
        print("Found mxr, loading it...")
        model = mgx.load(f"{file}.mxr", format="msgpack")
    elif os.path.isfile(f"{file}.onnx"):
        print("Parsing from onnx file...")
        model = mgx.parse_onnx(f"{file}.onnx", map_input_dims=shapes)
        model.compile(mgx.get_target("gpu"))
        print(f"Saving {name} model to mxr file...")
        mgx.save(model, f"{file}.mxr", format="msgpack")
    else:
        print(f"No {name} model found. Please download it and re-try.")
        os.exit(1)
    return model


def main():
    args = get_args()

    model_id = "stabilityai/stable-diffusion-2-1"
    print(f"Using {model_id}")

    print(
        f"Creating EulerDiscreteScheduler scheduler with {args.steps} steps..."
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id,
                                                       subfolder="scheduler")
    scheduler.set_timesteps(args.steps)

    print("Creating CLIPTokenizer tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    print("Load models...")
    vae = load_mgx_model("vae_decoder", {"latent_sample": [1, 4, 64, 64]})
    text_encoder = load_mgx_model("text_encoder", {"input_ids": [1, 77]})
    unet = load_mgx_model(
        "unet", {
            "sample": [1, 4, 64, 64],
            "encoder_hidden_states": [1, 77, 1024],
            "timestep": [1],
        })

    print("Tokenizing prompt...")
    text_input = tokenizer([args.prompt],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="np")
    print("Creating text embeddings for prompt...")
    text_embeddings = np.array(
        text_encoder.run({"input_ids": text_input.input_ids.astype(np.int32)
                          })[0]).astype(np.float32)

    print("Tokenizing negative prompt...")
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([args.negative_prompt],
                             padding="max_length",
                             max_length=max_length,
                             return_tensors="np")

    print("Creating text embeddings for negative prompt...")
    uncond_embeddings = np.array(
        text_encoder.run(
            {"input_ids":
             uncond_input.input_ids.astype(np.int32)})[0]).astype(np.float32)

    print(
        f"Creating random input data ({1}x{4}x{64}x{64}) (latents) with seed={args.seed}..."
    )
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=torch.manual_seed(args.seed),
    )

    print("Apply initial noise sigma")
    latents = latents * scheduler.init_noise_sigma

    print("Running denoising loop...")
    for t in tqdm(scheduler.timesteps):
        sample = scheduler.scale_model_input(latents,
                                             t).numpy().astype(np.float32)
        timestep = np.atleast_1d(t.numpy().astype(
            np.int64))  # convert 0D -> 1D

        noise_pred_uncond = np.array(
            unet.run({
                "sample": sample,
                "encoder_hidden_states": uncond_embeddings,
                "timestep": timestep
            })[0])

        noise_pred_text = np.array(
            unet.run({
                "sample": sample,
                "encoder_hidden_states": text_embeddings,
                "timestep": timestep
            })[0])

        # perform guidance
        noise_pred = noise_pred_uncond + args.scale * (noise_pred_text -
                                                       noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(torch.from_numpy(noise_pred), t,
                                 latents).prev_sample

    print("Scale denoised result...")
    latents = 1 / 0.18215 * latents

    print("Decode denoised result...")
    image = np.array(
        vae.run({"latent_sample": latents.numpy().astype(np.float32)})[0])

    print("Convert result to rgb image...")
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = np.transpose(image, (0, 2, 3, 1))
    images = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(images[0])

    filename = args.output if args.output else f"output_s{args.seed}_t{args.steps}.png"
    pil_image.save(filename)
    print(f"Image saved to {filename}")


if __name__ == "__main__":
    main()

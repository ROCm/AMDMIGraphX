import os
import sys
import pathlib
from PIL import Image
import time

from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer

import migraphx
import torch
from torch_migraphx.fx import MGXModule


class MGXSDXLPipeline:

    def __init__(self,
                 base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
                 denoising_steps=30,
                 guidance_scale=7.5,
                 vae_scaling_factor=0.18215,
                 device='cuda',
                 quantize_fp16=False,
                 exhaustive_tune=False,
                 seed=None):
        self.base_model_path = base_model_path
        self.denoising_steps = denoising_steps
        self.guidance_scale = guidance_scale
        self.vae_scaling_factor = vae_scaling_factor
        self.quantize_fp16 = quantize_fp16
        self.exhaustive_tune = exhaustive_tune
        self.device = device
        self.noise_gen = torch.Generator(
            device="cuda").manual_seed(seed) if seed else None

    def _load_onnx_or_pt(self, path, name, shapes, file_type, fp16=False):
        file = f"{path}/{name}/model.{file_type}"

        if not os.path.isfile(file):
            raise FileNotFoundError(file)

        print(f"Loading {name} model from {file}")
        if file_type == "onnx":
            print("Parsing from onnx file...")
            prog = migraphx.parse_onnx(file, map_input_dims=shapes)
            mgx_mod = MGXModule(prog,
                                prog.get_parameter_names(),
                                quantize_fp16=fp16,
                                exhaustive_tune=self.exhaustive_tune)
            return mgx_mod
        elif file_type == "pt":
            print("Found pt, loading it...")
            return torch.load(file)
        elif file_type == "mxr":
            print("Found mxr, loading it...")
            prog = migraphx.load(f"{file}.mxr", format="msgpack")
            # Compiled mxrs will have output buffers as part of parameters
            input_names = [
                "encoder_hidden_states", "timestep", "time_ids", "text_embeds",
                "sample"
            ]
            mgx_mod = MGXModule(prog,
                                input_names,
                                exhaustive_tune=self.exhaustive_tune)
        else:
            raise NotImplementedError(f"Unsupported file type: {file_type}")

    def load_models(self, model_path, file_type="onnx"):
        self.models = {}
        self.models["scheduler"] = EulerDiscreteScheduler.from_pretrained(
            self.base_model_path, subfolder="scheduler")
        self.models["scheduler"].set_timesteps(self.denoising_steps)

        self.models["tokenizer"] = CLIPTokenizer.from_pretrained(
            self.base_model_path, subfolder="tokenizer")

        self.models["tokenizer2"] = CLIPTokenizer.from_pretrained(
            self.base_model_path,
            use_safetensors=True,
            subfolder="tokenizer_2")

        self.models["clip"] = self._load_onnx_or_pt(model_path, "clip",
                                                    {"input_ids": [1, 77]},
                                                    file_type)
        self.models["clip2"] = self._load_onnx_or_pt(model_path, "clip2",
                                                     {"input_ids": [1, 77]},
                                                     file_type)
        self.models["unetxl"] = self._load_onnx_or_pt(
            model_path,
            "unetxl", {
                "sample": [2, 4, 128, 128],
                "encoder_hidden_states": [2, 77, 2048],
                "text_embeds": [2, 1280],
                "time_ids": [2, 6],
                "timestep": [1],
            },
            file_type,
            fp16=self.quantize_fp16)

        self.models["vae_decoder"] = self._load_onnx_or_pt(
            model_path, "vae", {"latent_sample": [1, 4, 128, 128]}, file_type)

        # Initialize hip events for timing
        self.events = {}
        for stage in ['clip', 'denoise', 'vae']:
            for marker in ['start', 'stop']:
                self.events[f"{stage}-{marker}"] = torch.cuda.Event(
                    enable_timing=True)

    def save_compiled_models(self, path, file_type="pt"):
        for name in ["clip", "clip2", "unetxl", "vae"]:
            model_path = pathlib.Path(f"{path}/{name}")
            model_path.mkdir(parents=True, exist_ok=True)
            file = model_path / f"model.{file_type}"
            model_name = "vae_decoder" if name == "vae" else name

            print(f"Saving {model_name} model to {file}")
            if file_type == "pt":
                torch.save(self.models[model_name], file, pickle_protocol=4)
            elif file_type == "mxr":
                migraphx.save(self.models[model_name].program,
                              file,
                              format="msgpack")
            else:
                raise NotImplementedError(
                    f"Unsupported file type: {file_type}")

    def initialize_latents(self, batch_size, unet_channels, latent_height,
                           latent_width):
        latents_dtype = torch.float32
        latents_shape = (batch_size, unet_channels, latent_height,
                         latent_width)
        latents = torch.randn(latents_shape,
                              device=self.device,
                              dtype=latents_dtype,
                              generator=self.noise_gen)
        latents = latents * self.models["scheduler"].init_noise_sigma
        return latents

    def encode_prompt(self,
                      prompt,
                      negative_prompt,
                      encoder="clip",
                      tokenizer="tokenizer",
                      pooled_outputs=False,
                      output_hidden_states=False):

        self.events["clip-start"].record()
        tokenizer_mod = self.models[tokenizer]
        encoder_mod = self.models[encoder]

        text_input_ids = tokenizer_mod(
            prompt,
            padding="max_length",
            max_length=tokenizer_mod.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        outs = encoder_mod(text_input_ids)
        # Outputs for clip2 are reversed for some reason??
        if encoder == "clip2":
            text_embeddings, hidden_states = outs[0].clone(), outs[1].clone()
        else:
            text_embeddings, hidden_states = outs[1].clone(), outs[0].clone()

        uncond_input_ids = tokenizer_mod(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer_mod.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        uncond_outs = encoder_mod(uncond_input_ids)
        if encoder == "clip2":
            uncond_embeddings, uncond_hidden_states = uncond_outs[0].clone(
            ), uncond_outs[1].clone()
        else:
            uncond_embeddings, uncond_hidden_states = uncond_outs[1].clone(
            ), uncond_outs[0].clone()

        text_embeddings = torch.cat([uncond_embeddings,
                                     text_embeddings]).to(dtype=torch.float16)

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            text_embeddings = torch.cat([uncond_hidden_states, hidden_states
                                         ]).to(dtype=torch.float16)

        self.events["clip-stop"].record()

        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    def _get_add_time_ids(self, original_size, crops_coords_top_left,
                          target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left +
                            target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def denoise_latent(self,
                       latents,
                       hidden_states,
                       time_ids,
                       text_embeds,
                       denoiser='unetxl',
                       timesteps=None,
                       guidance=7.5):

        assert guidance > 1.0, "Guidance has to be > 1.0"
        denoiser_mod = self.models[denoiser]
        scheduler_mod = self.models["scheduler"]

        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.models["scheduler"].timesteps
            
        self.events["denoise-start"].record()

        for step_index, timestep in enumerate(timesteps):
            latents = latents.to(torch.float32)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler_mod.scale_model_input(
                latent_model_input, timestep)

            timestep_float = timestep.float(
            ) if timestep.dtype != torch.float32 else timestep

            # MGXModule input ordering is encoder_hidden_states, timestep, time_ids, text_embeds, sample
            noise_pred = denoiser_mod(latent_model_input, timestep,
                                      hidden_states, text_embeds, time_ids)

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text -
                                                         noise_pred_uncond)

            latents = scheduler_mod.step(noise_pred, timestep,
                                         latents).prev_sample

        latents = 1. / self.vae_scaling_factor * latents

        self.events["denoise-stop"].record()
        return latents

    def decode_latent(self, latents):
        decoder = self.models["vae_decoder"]
        self.events["vae-start"].record()

        latents = latents.to(dtype=torch.float32)
        images = decoder(latents)

        self.events["vae-stop"].record()
        return images

    def save_image(self, images, image_path_dir):
        images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(
            0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
        for i in range(images.shape[0]):
            image_path = os.path.join(image_path_dir, f"output{i+1}.png")
            print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
            Image.fromarray(images[i]).save(image_path)

    def run(self, prompts, neg_prompts=None):
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        guidance = 5.0

        prompts = [prompts] if isinstance(prompts, str) else prompts
        self.batch_size = len(prompts)

        if neg_prompts is None:
            neg_prompts = ["" for _ in prompts]
        elif isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts]

        assert len(prompts) == len(neg_prompts)
        
        with torch.inference_mode(), torch.autocast("cuda"):

            latents = self.initialize_latents(batch_size=self.batch_size,
                                            unet_channels=4,
                                            latent_height=128,
                                            latent_width=128)
            
            torch.cuda.synchronize()
            self.e2e_tic = time.perf_counter()

            text_embeddings = self.encode_prompt(prompts,
                                                neg_prompts,
                                                encoder='clip',
                                                tokenizer='tokenizer',
                                                output_hidden_states=True)

            text_embeddings2, pooled_embeddings2 = self.encode_prompt(
                prompts,
                neg_prompts,
                encoder='clip2',
                tokenizer='tokenizer2',
                pooled_outputs=True,
                output_hidden_states=True)

            text_embeddings = torch.cat([text_embeddings, text_embeddings2],
                                        dim=-1)

            add_time_ids = self._get_add_time_ids(original_size,
                                                crops_coords_top_left,
                                                target_size,
                                                dtype=text_embeddings.dtype)
            add_time_ids = add_time_ids.repeat(self.batch_size, 1)
            add_time_ids = torch.cat([add_time_ids, add_time_ids],
                                    dim=0).to(self.device)

            latents = self.denoise_latent(latents=latents,
                                        hidden_states=text_embeddings,
                                        time_ids=add_time_ids,
                                        text_embeds=pooled_embeddings2,
                                        denoiser='unetxl',
                                        guidance=guidance)

            images = self.decode_latent(latents)
            
            torch.cuda.synchronize()
            self.e2e_toc = time.perf_counter()

        return images
    
    def print_summary(self):
        print('|------------|--------------|')
        print('| {:^10} | {:^12} |'.format('Module', 'Latency'))
        print('|------------|--------------|')
        print('| {:^10} | {:>9.2f} ms |'.format('CLIP', self.events['clip-start'].elapsed_time(self.events['clip-stop'])))
        print('| {:^10} | {:>9.2f} ms |'.format('UNet x '+str(self.denoising_steps), self.events['denoise-start'].elapsed_time(self.events['denoise-stop'])))
        print('| {:^10} | {:>9.2f} ms |'.format('VAE-Dec', self.events['vae-start'].elapsed_time(self.events['vae-stop'])))
        print('|------------|--------------|')
        print('| {:^10} | {:>9.2f} ms |'.format('Pipeline', (self.e2e_toc - self.e2e_tic)*1000.))
        print('|------------|--------------|')
        print('Throughput: {:.2f} image/s'.format(self.batch_size/(self.e2e_toc - self.e2e_tic)))

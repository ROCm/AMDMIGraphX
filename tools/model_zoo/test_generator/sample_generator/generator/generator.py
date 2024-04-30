#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#
#####################################################################################
import os
import onnxruntime as ort
import numpy as np
import torch

from ..utils import get_model_io, numpy_to_pb


def _inference(session, input_data_map, outputs, folder_name):
    input_pb_name = "input_{}.pb"
    output_pb_name = "output_{}.pb"
    os.makedirs(folder_name, exist_ok=True)
    for input_idx, (input_name,
                    input_data) in enumerate(input_data_map.items()):
        numpy_to_pb(input_name, input_data,
                    f"{folder_name}/{input_pb_name.format(input_idx)}")

    ort_result = session.run(outputs, input_data_map)
    output_data_map = {
        output_name: result_data
        for (output_name, result_data) in zip(outputs, ort_result)
    }
    for output_idx, (output_name,
                     result_data) in enumerate(output_data_map.items()):
        numpy_to_pb(output_name, result_data,
                    f"{folder_name}/{output_pb_name.format(output_idx)}")

    return output_data_map


def generate_test_dataset(model,
                          dataset,
                          output_folder_prefix=None,
                          sample_limit=None,
                          decode_limit=None):
    assert model.is_diffuser() == False, "Only base and decoder supported"
    output_path = f"{output_folder_prefix or 'generated'}/{dataset.name()}/{model.name()}"
    folder_name_prefix = f"{output_path}/test_data_set"

    # Model
    try:
        model_path = model.get_model(output_path)
    except Exception as e:
        print(f"Something went wrong:\n{e}\nSkipping model...")
        return
    inputs, outputs = get_model_io(model_path)

    sess = ort.InferenceSession(model_path)
    print(f"Creating {folder_name_prefix}s...")
    test_idx = 0
    for idx, data in enumerate(dataset):
        input_data_map = dataset.transform(inputs, data, model.preprocess)
        is_eos, decode_idx = False, 0
        while not is_eos and not (decode_limit and decode_limit <= decode_idx):
            folder_name = f"{folder_name_prefix}_{test_idx}"

            output_data_map = _inference(sess, input_data_map, outputs,
                                         folder_name)

            is_eos = not model.is_decoder() or model.decode_step(
                input_data_map, output_data_map)
            test_idx += 1
            decode_idx += 1

        if sample_limit and sample_limit - 1 <= idx:
            break


def generate_diffusion_data(model,
                            image_dataset,
                            prompt_dataset,
                            output_folder_prefix=None,
                            sample_limit=None,
                            decode_limit=None):
    assert model.is_diffuser() == True, "Only diffuser supported"
    output_path = f"{output_folder_prefix or 'generated'}/diffusion/{model.name()}"

    try:
        model_paths = model.get_model(output_path)
    except Exception as e:
        print(f"Something went wrong:\n{e}\nSkipping model...")
        return

    #### Text Encoder ####
    text_encoder_model_path = model_paths[0]
    text_encoder_inputs, text_encoder_outputs = get_model_io(
        text_encoder_model_path)
    text_encoder_sess = ort.InferenceSession(text_encoder_model_path)
    text_encoder_folder_name_prefix = f"{os.path.dirname(text_encoder_model_path)}/test_data_set"
    ######################

    #### VAE Encoder ####
    vae_encoder_model_path = model_paths[1]
    vae_encoder_inputs, vae_encoder_outputs = get_model_io(
        vae_encoder_model_path)
    vae_encoder_sess = ort.InferenceSession(vae_encoder_model_path)
    vae_encoder_folder_name_prefix = f"{os.path.dirname(vae_encoder_model_path)}/test_data_set"
    #####################

    #### UNet ####
    unet_model_path = model_paths[2]
    unet_inputs, unet_outputs = get_model_io(unet_model_path)
    unet_sess = ort.InferenceSession(unet_model_path)
    unet_folder_name_prefix = f"{os.path.dirname(unet_model_path)}/test_data_set"
    ##############

    #### VAE Decoder ####
    vae_decoder_model_path = model_paths[3]
    vae_decoder_inputs, vae_decoder_outputs = get_model_io(
        vae_decoder_model_path)
    vae_decoder_sess = ort.InferenceSession(vae_decoder_model_path)
    vae_decoder_folder_name_prefix = f"{os.path.dirname(vae_decoder_model_path)}/test_data_set"
    #####################

    test_idx = 0
    scale = 7.0
    seed = 42
    # TODO: get it from latent size
    sample_shape = (1, 4, 64, 64)
    # TODO from config?
    vae_scaling_factor = 0.18215
    noise = torch.randn(sample_shape, generator=torch.manual_seed(seed))
    for idx, (image, prompt) in enumerate(zip(image_dataset, prompt_dataset)):
        #### Text Encoder ####
        text_encoder_input_data_map = prompt_dataset.transform(
            text_encoder_inputs, prompt, model.text_preprocess)
        text_encoder_folder_name = f"{text_encoder_folder_name_prefix}_{test_idx}"
        text_encoder_output_data_map = _inference(
            text_encoder_sess,
            text_encoder_input_data_map,
            text_encoder_outputs,
            text_encoder_folder_name,
        )
        ######################

        #### VAE Encoder ####
        vae_encoder_input_data_map = image_dataset.transform(
            vae_encoder_inputs, image, model.image_preprocess)
        vae_encoder_folder_name = f"{vae_encoder_folder_name_prefix}_{test_idx}"
        vae_encoder_output_data_map = _inference(
            vae_encoder_sess,
            vae_encoder_input_data_map,
            vae_encoder_outputs,
            vae_encoder_folder_name,
        )
        #####################

        #### UNET ####
        model.scheduler.set_timesteps(decode_limit)
        latents = torch.from_numpy(
            vae_encoder_output_data_map["latent_sample"]) * vae_scaling_factor
        latents = model.scheduler.add_noise(latents, noise,
                                            model.scheduler.timesteps[:1])

        for step, t in enumerate(model.scheduler.timesteps):
            unet_folder_name = f"{unet_folder_name_prefix}_{test_idx*decode_limit + step}"

            latents_model_input = torch.concatenate([latents] * 2)
            latents_model_input = model.scheduler.scale_model_input(
                latents_model_input, t).numpy().astype(np.float32)
            timestep = np.atleast_1d(t.numpy().astype(np.int64))

            unet_input_data_map = {
                'sample':
                latents_model_input,
                'encoder_hidden_states':
                text_encoder_output_data_map['last_hidden_state'],
                'timestep':
                timestep
            }

            unet_output_data_map = _inference(
                unet_sess,
                unet_input_data_map,
                unet_outputs,
                unet_folder_name,
            )

            noise_pred_text, noise_pred_uncond = np.array_split(
                unet_output_data_map['out_sample'], 2)

            noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                      noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = model.scheduler.step(torch.from_numpy(noise_pred), t,
                                           latents).prev_sample

        ############

        #### VAE Decoder ####
        latents = 1 / vae_scaling_factor * latents
        vae_decoder_input_data_map = {
            "latent_sample": latents.numpy().astype(np.float32)
        }
        vae_decoder_folder_name = f"{vae_decoder_folder_name_prefix}_{test_idx}"
        vae_decoder_output_data_map = _inference(
            vae_decoder_sess,
            vae_decoder_input_data_map,
            vae_decoder_outputs,
            vae_decoder_folder_name,
        )
        #####################

        test_idx += 1
        if sample_limit and sample_limit - 1 <= idx:
            break

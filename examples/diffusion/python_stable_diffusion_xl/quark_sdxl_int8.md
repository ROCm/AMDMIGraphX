# Stable Diffusion XL

This version was tested with [rocm 6.0](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/rocm-6.0.0) revision.

## Console application

To run the console application, follow these steps below.

Setup python environment

```bash
# this will require the python venv to installed (e.g. apt install python3.8-venv)
python3 -m venv sd_venv
. sd_venv/bin/activate
```

Install dependencies

```bash
pip install -U pip
pip install -r torch_requirements.txt -r requirements.txt
```

Use MIGraphX Python Module

```bash
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

Get models with huggingface-cli

### Turbo version

```bash
huggingface-cli download stabilityai/sdxl-turbo text_encoder/model.onnx text_encoder_2/model.onnx text_encoder_2/model.onnx_data vae_decoder/model.onnx --local-dir models/sdxl-turbo/ --local-dir-use-symlinks False
```

### Get Quark SDXL Int8 from nas_share

```bash
cp -r /mnt/nas_share/migraphx/models/quark/SDXL-UNET-INT8-ONNX/quant_model_unet/ models/sdxl-turbo/unet/

mv models/sdxl-turbo/unet/sdxl.onnx models/sdxl-turbo/unet/model.onnx
```

### Running txt2img with UNet Int8 

```bash
python3 txt2img.py --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --seed 42 --output jungle_astro.jpg --pipeline-type sdxl-turbo --unet_int8
```

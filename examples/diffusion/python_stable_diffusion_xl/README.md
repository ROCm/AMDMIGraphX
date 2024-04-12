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
pip install -r requirements.txt
```

Use MIGraphX Python Module

```bash
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

Get models with huggingface-cli

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 vae_decoder/model.onnx --local-dir models/sdxl-1.0-base/ --local-dir-use-symlinks False
huggingface-cli download stabilityai/stable-diffusion-xl-1.0-tensorrt sdxl-1.0-base/clip.opt/model.onnx sdxl-1.0-base/clip2.opt/model.onnx sdxl-1.0-base/unetxl.opt/model.onnx sdxl-1.0-base/unetxl.opt/435d4c0a-2d32-11ee-8476-0242c0a80101 --local-dir models/ --local-dir-use-symlinks False
```
*Note: `models/sdxl-1.0-base` will be used in the scripts.*

Convert CLIP models to expose "hidden_state" as output.

```bash
# clip.opt
python clip_modifier.py -i models/sdxl-1.0-base/clip.opt/model.onnx -o models/sdxl-1.0-base/clip.opt.mod/model.onnx

# clip2.opt
python clip_modifier.py -i models/sdxl-1.0-base/clip2.opt/model.onnx -o models/sdxl-1.0-base/clip2.opt.mod/model.onnx
```

Run the text-to-image script with the following example prompt and seed:

```bash
python txt2img.py --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --seed 42 --output jungle_astro.jpg
```
*Note: The first run will compile the models and cache them to make subsequent runs faster.*

The result should look like this:

![example_output.jpg](./example_output.jpg)

## (Optional) Refiner

Note: requires `Console application` to work

Get models with huggingface-cli

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-1.0-tensorrt sdxl-1.0-refiner/unetxl.opt/model.onnx sdxl-1.0-refiner/unetxl.opt/6ed855ee-2d70-11ee-af8e-0242c0a80101 sdxl-1.0-refiner/unetxl.opt/6e186582-2d74-11ee-8aa7-0242c0a80102 --local-dir models/ --local-dir-use-symlinks False
```

Run the text-to-image script with the following example prompt and seed:

```bash
python txt2img.py --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --seed 42 --output refined_jungle_astro.jpg --refiner-onnx-model-path models/sdxl-1.0-refiner
```

## Gradio application

Note: requires `Console application` to work

Install gradio dependencies

```bash
pip install -r gradio_requirements.txt
```

Usage

```bash
python gradio_app.py -p "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```

This will load the models (which can take several minutes), and when the setup is ready, starts a server on `http://127.0.0.1:7860`.

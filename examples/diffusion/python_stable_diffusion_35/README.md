# Stable Diffusion 3.5

This version was tested with [rocm 6.4](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/rocm-6.4.0) revision.

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
pip install -r torch_requirements.txt
pip install -r requirements.txt
```

Use MIGraphX Python Module

```bash
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

Get models:

Make sure you have permission to download and use stabilityai/stable-diffusion-3.5-medium.
```bash
huggingface-cli login
```

Export the models to onnx. 

```bash
optimum-cli export onnx --model stabilityai/stable-diffusion-3.5-medium  models/sd35
```

Run the text-to-image script with the following example prompt and seed (optionally, you can change the batch size / number of images generated for that prompt)

```bash
python txt2img.py --prompt "a photograph of an astronaut riding a horse" --steps 50 --output astro_horse.jpg
```
> [!NOTE]
> The first run will compile the models and cache them to make subsequent runs faster. New batch sizes will result in the models re-compiling.*

The result should look like this:

![example_output.jpg](./example_output.jpg)

## Lower Memory Usage Pipeline
The entire pipeline is memory intensive, even when quantizing to fp16. The T5XXL encoder can be disabled alongside fp16 quantization to reduce total GPU memory usage to under 16G.

There will be a slight accuracy penalty when disabling T5XXL.
```bash
python txt2img.py --prompt "a photograph of an astronaut riding a horse" --steps 50 --skip-t5 --fp16=all --output astro_horse.jpg
```


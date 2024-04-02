## Build and Run Docker Image
```
docker build -t sdxl_perf .

docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=`pwd`:/workspace -v=/mnt/nas_share/migraphx/models:/models --entrypoint=/bin/bash -e HIP_FORCE_DEV_KERNARG=1 -e MIGRAPHX_MLIR_USE_SPECIFIC_OPS=dot,fused,attention  sdxl_perf
```

## Convert CLIP models

```bash
# clip.opt
python clip_modifier.py -i /models/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base/clip.opt/model.onnx /models/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base/clip.opt.mod/model.onnx

# clip2.opt
python clip_modifier.py -i /models/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base/clip2.opt/model.onnx /models/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base/clip2.opt.mod/model.onnx
```

## Run Image-to-Text script
```
python txt2img.py --base-model-path /models/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base/ --save-compiled
```

## Run pipeline using Torch-MIGraphX
```
python torch_migraphx/benchmark_sdxl.py
```
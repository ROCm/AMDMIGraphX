# MIGraphX Model Optimizer
This command line tool is meant to provide a simple way to optimize and serialize deep learning models for faster deployment. 

## Basic Usage
MIGraphX must first be built in order to use the model optimizer. Please refer to the top-level [README](/README.md) for instructions on how to build and install.


```
$ python3 optimize_model.py -h
```

```
usage: optimize_model.py [-h] [--nchw] [-t {gpu,cpu,ref}] [-d] [-o] [-f] [-i] [-j] [-p OUTPUT_PATH] model

Compile and serialize model offline for future deployment.

positional arguments:
  model                 Path to ONNX or TF Protobuf file

optional arguments:
  -h, --help            show this help message and exit
  --nchw                Treat Tensorflow format as nchw. (Default is nhwc)
  -t {gpu,cpu,ref}, --target {gpu,cpu,ref}
                        Compilation target
  -d, --disable_fast_math
                        Disable optimized math functions
  -o, --offload_copy_off
                        Disable implicit offload copying
  -f, --fp16            Quantize model in FP16 precision
  -i, --int8            Quantize model in INT8 precision
  -j, --json            Save program in JSON format. (Default is MsgPack format)
  -p OUTPUT_PATH, --output_path OUTPUT_PATH
                        Specify file name and/or path for model to be saved. (Default is
                        ./saved_models/<model_name>.<format>
```

## Targets
Models can be compiled to run on the CPU or GPU. The "ref" (reference) target uses CPU operators, but is primarily meant for checking correctness. To compile for the "cpu" target, MIGraphX needs to be built using the `-DMIGRAPHX_ENABLE_CPU=On` flag; i.e.: `CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIGRAPHX_ENABLE_CPU=On ..`.

## Optimizations
The compilation process performs multiple optimization passes on the input model to produce a more efficient network graph. In addition to the standard passes, two more optional optimizations can be turned on or off: `offload_copy` and `fast_math`. 

- `offload_copy`: For targets with offloaded memory (such as the gpu), this will insert instructions during compilation to copy the input parameters to the offloaded memory and to copy the final result from the offloaded memory back to main memory.
- `fast_math`: Optimize math functions to use faster approximate versions. There may be slight accuracy degredation when enabled.

### FP16 Quantization
Enabling FP16 quantization will convert data and weight tensors to 16-bit floating points values to reduce the model's size and to improve performance on hardware that supports half-precision operations. 

### INT8 Quantization
Enabling INT8 quantization will similarly convert data and weight tensors to 8-bit integer values. Calibration is not currently supported by this tool, but can be implemented using either of MIGraphX's Python or C++ APIs. Please refer to [this example](../cpp_api_inference/README.md) to see how calibration is set up with the C++ API. 

## Serialization
Finally, once all of the optimizations and compilation have taken place, the model can be serialized to either Messagepack (.msgpack) or JSON (.json) format for future use. The default format is Messagepack, but can be switched to JSON using the `-j, --json` option. Once a model has been compiled and serialized, it can be quickly loaded and executed to perform inference without the latency of re-compiling. 

By default, a model, e.g. `model_name.onnx`, passed into `optimize_model.py` will be saved as `model_name.msgpack`, but an alternate name can be supplied with the `-p, --output_path` option. The default path where the model will be saved is `/AMDMIGraphX/examples/python_model_optimizer/saved_models/`, but may also be manually selected using the `-p, --output_path` option. The `OUTPUT_PATH` argument following `-p, --output_path` may be a full path and file name, just a path (default file name will be used), or just a file name (default path will be used). 
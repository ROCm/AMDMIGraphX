# AMD MIGraphX Autotune Perf
## Instructions
First ensure MIGraphX is built so that `migraphx-driver` is available (e.g. `./build/bin/migraphx-driver`). Refer to MIGraphX instructions at the root directory for build steps.
The autotune script sweeps a curated set of MIGraphX environment-variable knobs on top of `migraphx-driver perf` and reports the fastest configuration for a given model. Therefore, an onnx file is required argument.
Example usage is below:
```
python autotune_perf.py --driver ./build/bin/migraphx-driver perf --onnx [path to onnx_file] --gpu
```

The same model used by `examples/vision/python_resnet50` can be fetched directly from the ONNX model zoo:
```
wget https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
python autotune_perf.py --driver ./build/bin/migraphx-driver perf --onnx resnet50-v2-7.onnx --gpu
```

The output of the script is a ranked table of configurations with the fastest row marked `<-- best`, and a sourceable `<model>.tune` file containing one `export` line per env var in the winning row.

By default the script tries the baseline, each knob in isolation, and a small set of curated multi-knob combinations. Pass `--no-combos` to skip the combinations (faster, but cannot find wins that require multiple knobs together).

The `migraphx-driver` binary is located in the following order: `--driver PATH`, then `$MIGRAPHX_DRIVER`, then `./bin/migraphx-driver` under the current working directory if executable, then `migraphx-driver` on `$PATH`.

By default the winning configuration is written to `<model>.tune` next to the model. Use `-o [path]` (or `--output [path]`) to write it elsewhere.

The generated config file can be sourced before subsequent `migraphx-driver` or application runs to pick up the tuned environment:
```
source resnet50-v2-7.onnx.tune
./build/bin/migraphx-driver perf --onnx resnet50-v2-7.onnx --gpu
```

Any flags after the script's own options are forwarded verbatim to `migraphx-driver perf`, so quantization, batch size, input dims, and other driver options work as usual (e.g. `--fp16`, `--batch 4`, `--input-dim @input 1 3 224 224`).

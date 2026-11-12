# Dynamic model examples

These examples download ONNX models, compile them once with symbolic dimensions,
and run the same compiled MIGraphX program with multiple runtime shapes.

- [Tiny-Llama](./tiny_llama): symbolic prompt lengths from 1 through 64.
- [Qwen](./qwen): symbolic prompt lengths from 1 through 64.

Set `MIGRAPHX_DRIVER` if `migraphx-driver` is not on `PATH`:

```bash
export MIGRAPHX_DRIVER=/path/to/migraphx-driver
```

Each model directory contains its own setup and execution instructions.
Downloaded models and compiled `.mxr` files are generated locally and should
not be committed.

# External Weights as Parameters

This example demonstrates how to swap ONNX external weight files at runtime
without re-parsing or re-compiling the model graph.

## Overview

Normally, `parse_onnx` reads external weight files (`.bin`) and bakes them into
the program as constants. Changing weights requires re-parsing and re-compiling.

With `external_weights_as_parameters=True`, the weights become program
parameters instead of constants. You can then:

1. **Parse once** -- no weight file I/O at parse time
2. **Compile once** -- shapes are known, values don't matter yet
3. **Load weights from any directory** -- `load_external_weights(prog, dir)`
4. **Swap freely** -- point to a new directory and call `load_external_weights` again

## Quick start

Generate a test model with two weight sets, then run the example:

```bash
python3 generate_test_model.py
python3 weight_params_example.py test_model/model.onnx test_model/weights_v1 test_model/weights_v2
```

`generate_test_model.py` creates:
- `test_model/model.onnx` -- a Conv model referencing external `weights.bin`
- `test_model/weights_v1/weights.bin` -- weights = 1.0, bias = 0.0
- `test_model/weights_v2/weights.bin` -- weights = 2.0, bias = 1.0

The example script will:
- Parse and compile the model once
- Load weights from each directory in turn
- Run inference with dummy input data
- Verify that different weights produce different outputs

## CLI equivalent

The MIGraphX driver also supports this via the `--weight-params` flag:

```bash
migraphx-driver read model.onnx --weight-params
```

# External Weights as Parameters (MXR Baking)

This example demonstrates how to create multiple self-contained MXR programs
from a single ONNX model by baking in different weight sets -- all without
re-parsing or re-compiling.

## Overview

Normally, `parse_onnx` reads external weight files (`.bin`) and bakes them into
the program as constants. Changing weights requires re-parsing and re-compiling.

With `external_weights_as_parameters=True`, the weights become program
parameters. You can then:

1. **Parse once** -- no weight file I/O at parse time
2. **Compile once** -- shapes are known, values don't matter yet
3. **Save the template** -- reuse without re-parse/re-compile
4. **Bake weights** -- `create_program_with_weights(prog, dir)` produces a new self-contained program
5. **Save baked MXR** -- deploy the result with weights built in

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
- Parse and compile the model once (producing a template)
- Bake weights from each directory into separate programs
- Save each as an MXR
- Verify that different weights produce different outputs

## CLI equivalent

The MIGraphX driver also supports the template-parsing step via `--weight-params`:

```bash
migraphx-driver read model.onnx --weight-params
```

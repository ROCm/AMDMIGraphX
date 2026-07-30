# External Weights (MXR Encoding)

This example demonstrates how to create multiple self-contained MXR programs
from a single ONNX model by encoding in different weight sets -- all without
re-parsing or re-compiling.

## Overview

Normally, `parse_onnx` reads external weight files (`.bin`) and loads them into
the program as constants. Changing weights requires re-parsing and re-compiling.

With `keep_weights_external=True`, the weights are kept external as
`external_weight` ops recorded in the IR. You can then:

1. **Parse once** -- no weight file I/O at parse time
2. **Compile once** -- shapes are known, values don't matter yet
3. **Save the template** -- reuse without re-parse/re-compile
4. **Encode weights** -- `replace_onnx_external_weights(prog, dir, target)` produces a new self-contained program
5. **Save encoded MXR** -- deploy the result with weights built in

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
- Encode weights from each directory into separate programs
- Save each as an MXR
- Verify that different weights produce different outputs

## ResNet50 example

For a real-world model, use the ResNet50 external-weights variant:

```bash
# 1. Get resnet50_v1.onnx from the ONNX model zoo
# 2. Convert to external weights format
python3 ../../convert_to_external_weights.py

# 3. Run the encoding example (creates original + perturbed MXRs)
python3 resnet50_gpu_encode_test.py resnet50_v1_external.onnx .
```

This parses + compiles ResNet50 once, then stamps out two MXRs with different
weight sets (original and noise-perturbed) without any recompilation.

## CLI equivalent

The MIGraphX driver supports the template-parsing step via `--weight-params`:

```bash
migraphx-driver read model.onnx --weight-params
migraphx-driver compile model.onnx --weight-params --gpu -o template.mxr
```

It can also encode a weight set during `compile` with `--encode-weights <dir>`,
either straight from the ONNX model or from a previously-saved template `.mxr`:

```bash
# Encode from the ONNX model
migraphx-driver compile model.onnx --weight-params --gpu \
    --encode-weights test_model/weights_v1 -o model_v1.mxr

# Or stamp a weight set into an existing compiled template
migraphx-driver compile template.mxr --gpu \
    --encode-weights test_model/weights_v1 -o model_v1.mxr
```

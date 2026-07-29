#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
"""
Example: Baking external weights into self-contained MXR programs.

This demonstrates the --weight-params / keep_weights_external feature
combined with replace_onnx_external_weights to produce MXR files with different
weight sets baked in -- all from a single parse + compile.

Typical use cases:
  - Generating deployment-ready MXRs for multiple fine-tuned variants
  - A/B testing different weight checkpoints
  - Offline baking of LoRA adapter variants

Directory layout assumed:
  model.onnx              <-- ONNX graph (references external weight files)
  weights_v1/
    weights.bin            <-- first set of weights
  weights_v2/
    weights.bin            <-- second set of weights (same shapes, different values)
"""

import sys
import numpy as np

import migraphx


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.onnx> <weights_dir> [weights_dir_2 ...]")
        print()
        print("  model.onnx    : ONNX model with external weight files")
        print("  weights_dir   : directory containing the .bin weight files")
        print("  weights_dir_2 : optional second directory to demonstrate swapping")
        sys.exit(1)

    model_path = sys.argv[1]
    weight_dirs = sys.argv[2:]

    # ------------------------------------------------------------------
    # Step 1: Parse with keep_weights_external=True
    #
    # Weights are kept external as external_weight ops in the IR (not baked-in
    # constants). No weight file I/O happens at parse time.
    # ------------------------------------------------------------------
    print(f"Parsing {model_path} with weights kept external...")
    template = migraphx.parse_onnx(model_path, keep_weights_external=True)

    param_shapes = template.get_parameter_shapes()
    print(f"Template has {len(param_shapes)} parameters:")
    for name, shape in param_shapes.items():
        print(f"  {name}: {shape}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Compile once
    #
    # The compiler knows the shapes of all parameters (including weights)
    # so it can optimize the graph. Actual weight values don't matter yet.
    # ------------------------------------------------------------------
    print("Compiling template (this only happens once)...")
    template.compile(migraphx.get_target("ref"))
    print("Compilation done.\n")

    # ------------------------------------------------------------------
    # Step 3: Save the template MXR (optional)
    #
    # This lets you skip parse+compile next time.
    # ------------------------------------------------------------------
    template_mxr = "template.mxr"
    migraphx.save(template, template_mxr)
    print(f"Saved template MXR: {template_mxr}\n")

    # ------------------------------------------------------------------
    # Step 4: Bake weights from each directory into separate programs
    #
    # replace_onnx_external_weights copies the template and replaces each
    # external_weight op with a literal read from the specified directory.
    # The result is a self-contained program you can save or run directly.
    # ------------------------------------------------------------------
    outputs = []
    for i, weight_dir in enumerate(weight_dirs):
        print(f"--- Baking weights from: {weight_dir} ---")
        baked = migraphx.replace_onnx_external_weights(template, weight_dir, migraphx.get_target("ref"))

        baked_params = baked.get_parameter_shapes()
        print(f"  Baked program has {len(baked_params)} parameters (weights gone):")
        for name, shape in baked_params.items():
            print(f"    {name}: {shape}")

        # Save baked MXR
        mxr_path = f"baked_v{i+1}.mxr"
        migraphx.save(baked, mxr_path)
        print(f"  Saved: {mxr_path}")

        # Run with dummy input
        all_params = {}
        for name, shape in baked_params.items():
            lens = shape.lens()
            dummy_input = np.ones(lens, dtype=np.float32)
            all_params[name] = migraphx.argument(dummy_input)

        results = baked.run(all_params)
        output = np.array(results[0])
        print(f"  Output shape: {output.shape}, sum: {output.sum():.4f}")
        print()
        outputs.append(output)

    # Verify different weights produce different outputs
    if len(outputs) >= 2:
        if np.array_equal(outputs[0], outputs[1]):
            print("WARNING: Outputs are identical -- weight baking may not have worked!")
            sys.exit(1)
        else:
            print("SUCCESS: Different weights produced different baked programs.")


if __name__ == "__main__":
    main()

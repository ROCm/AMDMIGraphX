"""
Example: Baking external weights into self-contained MXR programs.

This demonstrates the --weight-params / external_weights_as_parameters feature
combined with create_program_with_weights to produce MXR files with different
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
import os
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
    # Step 1: Parse with external_weights_as_parameters=True
    #
    # Weights become program parameters (not baked-in constants).
    # No weight file I/O happens at parse time.
    # ------------------------------------------------------------------
    print(f"Parsing {model_path} with weights as parameters...")
    template = migraphx.parse_onnx(model_path, external_weights_as_parameters=True)

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
    # create_program_with_weights copies the template and replaces weight
    # parameters with literals read from the specified directory.
    # The result is a self-contained program you can save or run directly.
    # ------------------------------------------------------------------
    outputs = []
    for i, weight_dir in enumerate(weight_dirs):
        print(f"--- Baking weights from: {weight_dir} ---")
        baked = migraphx.create_program_with_weights(template, weight_dir)

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

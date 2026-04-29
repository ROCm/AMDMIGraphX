"""
Example: Swapping external weights at runtime without re-parsing or re-compiling.

This demonstrates the --weight-params / external_weights_as_parameters feature.

Typical use cases:
  - Serving multiple fine-tuned variants of the same architecture
  - A/B testing different weight checkpoints
  - LoRA adapter swapping

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
        print("  weights_dir_2 : optional second directory to demonstrate hot-swapping")
        sys.exit(1)

    model_path = sys.argv[1]
    weight_dirs = sys.argv[2:]

    # ----------------------------------------------------------------
    # Step 1: Parse with external_weights_as_parameters=True
    #
    # This skips reading the weight files entirely during parse.
    # The weights become program parameters (like inputs) instead of
    # baked-in constants.
    # ----------------------------------------------------------------
    print(f"Parsing {model_path} with weights as parameters...")
    prog = migraphx.parse_onnx(model_path, external_weights_as_parameters=True)

    # Inspect what parameters the program has.
    # You'll see both the regular inputs AND the weight parameters.
    param_shapes = prog.get_parameter_shapes()
    print(f"Program has {len(param_shapes)} parameters:")
    for name, shape in param_shapes.items():
        print(f"  {name}: {shape}")
    print()

    # ----------------------------------------------------------------
    # Step 2: Compile once
    #
    # The compiler knows the shapes of all parameters (including weights)
    # so it can optimize the graph. The actual weight *values* don't
    # matter at compile time.
    # ----------------------------------------------------------------
    print("Compiling (this only happens once)...")
    prog.compile(migraphx.get_target("ref"))
    print("Compilation done.\n")

    # ----------------------------------------------------------------
    # Step 3: Load weights and run
    #
    # load_external_weights reads the .bin files from a directory and
    # returns a dict of {param_name: argument} for all weight parameters.
    # You merge this with your input data and call prog.run().
    # ----------------------------------------------------------------
    outputs = []
    for weight_dir in weight_dirs:
        print(f"--- Loading weights from: {weight_dir} ---")
        weight_params = migraphx.load_external_weights(prog, weight_dir)

        all_params = dict(weight_params)
        for name, shape in param_shapes.items():
            if name not in weight_params:
                lens = shape.lens()
                dtype = np.float32
                dummy_input = np.ones(lens, dtype=dtype)
                all_params[name] = migraphx.argument(dummy_input)
                print(f"  Input '{name}': shape={lens} (dummy data)")

        results = prog.run(all_params)
        output = np.array(results[0])
        print(f"  Output shape: {output.shape}, sum: {output.sum():.4f}")
        print()
        outputs.append(output)

    # Verify different weights produce different outputs
    if len(outputs) >= 2:
        if np.array_equal(outputs[0], outputs[1]):
            print("WARNING: Outputs are identical -- weight swap may not have worked!")
            sys.exit(1)
        else:
            print("SUCCESS: Different weights produced different outputs.")


if __name__ == "__main__":
    main()

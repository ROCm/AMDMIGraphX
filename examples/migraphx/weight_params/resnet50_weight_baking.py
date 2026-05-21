"""
Example: Baking different weight sets into ResNet50 MXRs.

Prerequisites:
  1. Download resnet50_v1.onnx (e.g. from the ONNX model zoo)
  2. Run convert_to_external_weights.py to produce:
       resnet50_v1_external.onnx
       resnet50_v1_external.weights

This script:
  - Parses the external-weights ONNX once (weights as parameters)
  - Compiles once
  - Creates two baked MXRs:
      * One with the original weights
      * One with perturbed weights (adds noise)
  - Verifies the two produce different outputs

Usage:
  python3 resnet50_weight_baking.py <resnet50_v1_external.onnx> <weights_dir>

  where <weights_dir> contains resnet50_v1_external.weights
"""

import sys
import os
import shutil
import numpy as np

import migraphx


def perturb_weights(src_dir, dst_dir, weight_filename, scale=0.01):
    """Copy weight file to dst_dir with random noise added."""
    os.makedirs(dst_dir, exist_ok=True)
    src_path = os.path.join(src_dir, weight_filename)
    dst_path = os.path.join(dst_dir, weight_filename)

    data = np.fromfile(src_path, dtype=np.uint8)
    floats = data.view(np.float32).copy()
    rng = np.random.default_rng(42)
    noise = rng.normal(0, scale, size=floats.shape).astype(np.float32)
    floats += noise
    floats.view(np.uint8).tofile(dst_path)

    print(f"  Created perturbed weights: {dst_path}")
    print(f"  Original size: {os.path.getsize(src_path):,} bytes")
    print(f"  Noise scale: {scale}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <resnet50_v1_external.onnx> <weights_dir>")
        print()
        print("  weights_dir should contain resnet50_v1_external.weights")
        sys.exit(1)

    model_path = sys.argv[1]
    weights_dir = sys.argv[2]

    weight_filename = "resnet50_v1_external.weights"
    weight_path = os.path.join(weights_dir, weight_filename)
    if not os.path.exists(weight_path):
        print(f"ERROR: {weight_path} not found")
        sys.exit(1)

    # Create a perturbed weights directory
    perturbed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet50_perturbed")
    print("Creating perturbed weight variant...")
    perturb_weights(weights_dir, perturbed_dir, weight_filename)
    print()

    # Step 1: Parse with weights as parameters
    print(f"Parsing {model_path} with weights as parameters...")
    template = migraphx.parse_onnx(model_path, external_weights_as_parameters=True)

    param_shapes = template.get_parameter_shapes()
    weight_params = [n for n in param_shapes if n != "data"]
    print(f"  Total parameters: {len(param_shapes)}")
    print(f"  Weight parameters: {len(weight_params)}")
    print(f"  Input parameters: {[n for n in param_shapes if n not in weight_params]}")
    print()

    # Step 2: Compile once
    print("Compiling template (one-time cost)...")
    template.compile(migraphx.get_target("ref"))
    print("  Done.\n")

    # Step 3: Save template MXR
    template_path = "resnet50_template.mxr"
    migraphx.save(template, template_path)
    print(f"Saved template: {template_path} ({os.path.getsize(template_path):,} bytes)\n")

    # Step 4: Bake original weights
    print(f"--- Baking original weights from: {weights_dir} ---")
    target = migraphx.get_target("ref")
    baked_original = migraphx.create_program_with_weights(template, weights_dir, target)
    orig_params = baked_original.get_parameter_shapes()
    print(f"  Baked program parameters: {list(orig_params.keys())}")

    migraphx.save(baked_original, "resnet50_baked_original.mxr")
    print(f"  Saved: resnet50_baked_original.mxr")

    # Run with dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result_orig = baked_original.run({"data": migraphx.argument(dummy_input)})
    output_orig = np.array(result_orig[0])
    print(f"  Output: shape={output_orig.shape}, top-5 indices={np.argsort(output_orig[0])[-5:][::-1]}")
    print()

    # Step 5: Bake perturbed weights
    print(f"--- Baking perturbed weights from: {perturbed_dir} ---")
    baked_perturbed = migraphx.create_program_with_weights(template, perturbed_dir, target)

    migraphx.save(baked_perturbed, "resnet50_baked_perturbed.mxr")
    print(f"  Saved: resnet50_baked_perturbed.mxr")

    result_pert = baked_perturbed.run({"data": migraphx.argument(dummy_input)})
    output_pert = np.array(result_pert[0])
    print(f"  Output: shape={output_pert.shape}, top-5 indices={np.argsort(output_pert[0])[-5:][::-1]}")
    print()

    # Verify outputs differ
    if np.array_equal(output_orig, output_pert):
        print("WARNING: Outputs are identical -- something went wrong!")
        sys.exit(1)
    else:
        diff = np.abs(output_orig - output_pert).mean()
        print(f"SUCCESS: Outputs differ (mean abs diff: {diff:.6f})")
        print("  Two MXRs produced from one template, no recompilation needed.")


if __name__ == "__main__":
    main()

"""
Example: Baking different weight sets into ResNet50 MXRs on GPU.

Compiles with offload_copy=False and runs inference with GPU buffers.
Supports --fp16 flag to quantize the model to half precision.

Prerequisites:
  1. Download resnet50_v1.onnx (e.g. from the ONNX model zoo)
  2. Run convert_to_external_weights.py to produce:
       resnet50_v1_external.onnx
       resnet50_v1_external.weights

Usage:
  python3 resnet50_gpu_bake_test.py <resnet50_v1_external.onnx> <weights_dir> [--fp16]

  where <weights_dir> contains resnet50_v1_external.weights
"""

import sys
import os
import argparse
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


def run_on_gpu(prog, input_data):
    """Run a program on GPU, allocating all required buffers."""
    params = prog.get_parameter_shapes()
    run_params = {}
    for name, shape in params.items():
        if name == "data":
            run_params[name] = migraphx.to_gpu(migraphx.argument(input_data))
        else:
            run_params[name] = migraphx.allocate_gpu(shape)

    result = prog.run(run_params)
    return np.array(migraphx.from_gpu(result[0]))


def main():
    parser = argparse.ArgumentParser(
        description="ResNet50 GPU weight baking test"
    )
    parser.add_argument("model_path", help="Path to resnet50_v1_external.onnx")
    parser.add_argument("weights_dir", help="Directory containing resnet50_v1_external.weights")
    parser.add_argument("--fp16", action="store_true", help="Quantize model to fp16")
    args = parser.parse_args()

    model_path = args.model_path
    weights_dir = args.weights_dir
    use_fp16 = args.fp16

    weight_filename = "resnet50_v1_external.weights"
    weight_path = os.path.join(weights_dir, weight_filename)
    if not os.path.exists(weight_path):
        print(f"ERROR: {weight_path} not found")
        sys.exit(1)

    precision = "fp16" if use_fp16 else "fp32"
    print("=" * 60)
    print(f"ResNet50 GPU Weight Baking Test ({precision}, offload_copy=False)")
    print("=" * 60)

    # Create perturbed weights directory
    perturbed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet50_perturbed")
    print("\nCreating perturbed weight variant...")
    perturb_weights(weights_dir, perturbed_dir, weight_filename)
    print()

    # Step 1: Parse with weights as parameters
    print(f"[1] Parsing {model_path} with weights as parameters...")
    template = migraphx.parse_onnx(model_path, external_weights_as_parameters=True)

    param_shapes = template.get_parameter_shapes()
    weight_params = [n for n in param_shapes if n != "data"]
    print(f"    Total parameters: {len(param_shapes)}")
    print(f"    Weight parameters: {len(weight_params)}")
    print(f"    Input parameters: {[n for n in param_shapes if n not in weight_params]}")
    print()

    # Step 1.5: Quantize to fp16 if requested
    if use_fp16:
        print("[1.5] Quantizing to fp16...")
        migraphx.quantize_fp16(template)
        print("    Done.\n")

    # Step 2: Compile for GPU with offload_copy=False
    print("[2] Compiling template for GPU (offload_copy=False)...")
    gpu_target = migraphx.get_target("gpu")
    template.compile(gpu_target, offload_copy=False)
    print("    Done.")
    params_after_compile = template.get_parameter_shapes()
    print(f"    Parameters after compile: {len(params_after_compile)}")
    print()

    # Step 3: Save template MXR
    suffix = f"_{precision}"
    template_path = f"resnet50_gpu_template{suffix}.mxr"
    migraphx.save(template, template_path)
    print(f"[3] Saved template: {template_path} ({os.path.getsize(template_path):,} bytes)\n")

    # Step 4: Bake original weights
    print(f"[4] Baking original weights from: {weights_dir}")
    baked_original = migraphx.create_program_with_weights(template, weights_dir, gpu_target)
    orig_params = baked_original.get_parameter_shapes()
    print(f"    Baked program parameters: {len(orig_params)}")
    print(f"    Remaining params: {list(orig_params.keys())[:5]}...")

    baked_orig_path = f"resnet50_gpu_baked_original{suffix}.mxr"
    migraphx.save(baked_original, baked_orig_path)
    print(f"    Saved: {baked_orig_path}")

    # Run with dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output_orig = run_on_gpu(baked_original, dummy_input)
    print(f"    Output: shape={output_orig.shape}, top-5 indices={np.argsort(output_orig[0])[-5:][::-1]}")
    print()

    # Step 5: Bake perturbed weights
    print(f"[5] Baking perturbed weights from: {perturbed_dir}")
    baked_perturbed = migraphx.create_program_with_weights(template, perturbed_dir, gpu_target)

    baked_pert_path = f"resnet50_gpu_baked_perturbed{suffix}.mxr"
    migraphx.save(baked_perturbed, baked_pert_path)
    print(f"    Saved: {baked_pert_path}")

    output_pert = run_on_gpu(baked_perturbed, dummy_input)
    print(f"    Output: shape={output_pert.shape}, top-5 indices={np.argsort(output_pert[0])[-5:][::-1]}")
    print()

    # Verify outputs differ
    if np.array_equal(output_orig, output_pert):
        print("FAIL: Outputs are identical -- something went wrong!")
        sys.exit(1)
    else:
        diff = np.abs(output_orig - output_pert).mean()
        print(f"PASS: Outputs differ (mean abs diff: {diff:.6f})")
        print(f"  Two {precision} GPU MXRs produced from one template, no recompilation needed.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

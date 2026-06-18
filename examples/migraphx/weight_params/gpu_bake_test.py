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
Simple GPU validation test for replace_external_weights.

Tests that baking lowers the inserted @literal instructions to gpu::literal
on a GPU-compiled program (without finalizing during the bake).

Steps:
  1. Generate a tiny model with external weights (matmul)
  2. Parse with keep_weights_external=True
  3. Compile for GPU
  4. Bake weights via replace_external_weights with GPU target
  5. Run inference on GPU and compare to a reference computation

Usage:
  python3 gpu_bake_test.py
"""

import os
import sys
import tempfile
import numpy as np

try:
    import migraphx
except ImportError:
    sys.exit("migraphx python module not found -- build with Python bindings enabled")

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    sys.exit("onnx python package required: pip install onnx")


def create_test_model(tmp_dir):
    """Create a simple MatMul model with external weights."""
    weight_shape = [4, 4]
    weight_nbytes = int(np.prod(weight_shape)) * 4  # float32

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    # External weight tensor
    weight_tensor = onnx.TensorProto()
    weight_tensor.name = "W"
    weight_tensor.data_type = TensorProto.FLOAT
    weight_tensor.dims.extend(weight_shape)
    weight_tensor.data_location = TensorProto.EXTERNAL
    weight_tensor.external_data.add(key="location", value="weights.bin")
    weight_tensor.external_data.add(key="offset", value="0")
    weight_tensor.external_data.add(key="length", value=str(weight_nbytes))

    matmul_node = helper.make_node("MatMul", ["input", "W"], ["output"])

    graph = helper.make_graph(
        [matmul_node], "gpu_bake_test", [X], [Y], initializer=[weight_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    model_path = os.path.join(tmp_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())

    return model_path, weight_shape


def write_weights(directory, weight_data):
    """Write weight array as raw binary weights.bin."""
    os.makedirs(directory, exist_ok=True)
    weight_path = os.path.join(directory, "weights.bin")
    with open(weight_path, "wb") as f:
        f.write(weight_data.astype(np.float32).tobytes())
    return directory


def main():
    print("=" * 60)
    print("GPU Weight Baking Validation Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create model
        print("\n[1] Creating test model (MatMul with external weights)...")
        model_path, weight_shape = create_test_model(tmp_dir)
        print(f"    Model: {model_path}")
        print(f"    Weight shape: {weight_shape}")

        # 2. Create weight file -- identity matrix so output == input
        W_identity = np.eye(4, dtype=np.float32)
        weights_dir = write_weights(os.path.join(tmp_dir, "weights"), W_identity)
        print(f"    Weights: identity matrix at {weights_dir}/weights.bin")

        # 3. Parse with keep_weights_external
        print("\n[2] Parsing ONNX with keep_weights_external=True...")
        prog = migraphx.parse_onnx(
            model_path, keep_weights_external=True
        )
        params = prog.get_parameter_shapes()
        print(f"    Parameters: {list(params.keys())}")
        # With keep_weights_external, 'W' is kept as an external_weight op in the
        # IR (not a parameter); only the real 'input' is a parameter.
        assert "W" not in params, "Weight 'W' should not be a parameter"
        assert "input" in params, "'input' should be a parameter"

        # 4. Compile for GPU (offload_copy=False so inputs must be GPU buffers)
        print("\n[3] Compiling for GPU target (offload_copy=False)...")
        gpu_target = migraphx.get_target("gpu")
        prog.compile(gpu_target, offload_copy=False)
        print("    Compilation successful.")
        params_after_compile = prog.get_parameter_shapes()
        print(f"    Parameters after compile: {list(params_after_compile.keys())}")

        # 5. Bake weights
        print("\n[4] Baking weights (replace_external_weights with GPU target)...")
        baked = migraphx.replace_external_weights(prog, weights_dir, gpu_target)
        baked_params = baked.get_parameter_shapes()
        print(f"    Baked program parameters: {list(baked_params.keys())}")
        assert "W" not in baked_params, "Weight 'W' should no longer be a parameter"
        assert "input" in baked_params, "'input' should still be a parameter"
        print("    Weight parameter successfully removed.")

        # Baking does not finalize, so materialize device buffers before running
        # in-process (a saved MXR is finalized automatically on load instead).
        baked.finalize(gpu_target)

        # 6. Run inference (must provide all GPU buffers with offload_copy=False)
        print("\n[5] Running inference on GPU...")
        input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        # Allocate all required parameters (input + output buffers)
        run_params = {}
        for pname, pshape in baked_params.items():
            if pname == "input":
                run_params[pname] = migraphx.to_gpu(migraphx.argument(input_data))
            else:
                run_params[pname] = migraphx.allocate_gpu(pshape)

        print(f"    Run parameters: {list(run_params.keys())}")
        result = baked.run(run_params)
        output = np.array(migraphx.from_gpu(result[0]))
        print(f"    Input:    {input_data}")
        print(f"    Output:   {output}")

        # With identity weights, output should equal input
        expected = input_data @ W_identity
        print(f"    Expected: {expected}")

        if np.allclose(output, expected, atol=1e-5):
            print("\n    PASS: Output matches expected result!")
        else:
            print(f"\n    FAIL: Output mismatch! Max diff = {np.max(np.abs(output - expected))}")
            sys.exit(1)

        # 7. Test with different weights to confirm baking actually changes behavior
        print("\n[6] Testing with different weights (all 0.5)...")
        W_half = np.full(weight_shape, 0.5, dtype=np.float32)
        weights_dir2 = write_weights(os.path.join(tmp_dir, "weights2"), W_half)

        baked2 = migraphx.replace_external_weights(prog, weights_dir2, gpu_target)
        baked2.finalize(gpu_target)
        baked2_params = baked2.get_parameter_shapes()
        run_params2 = {}
        for pname, pshape in baked2_params.items():
            if pname == "input":
                run_params2[pname] = migraphx.to_gpu(migraphx.argument(input_data))
            else:
                run_params2[pname] = migraphx.allocate_gpu(pshape)

        result2 = baked2.run(run_params2)
        output2 = np.array(migraphx.from_gpu(result2[0]))

        expected2 = input_data @ W_half
        print(f"    Input:    {input_data}")
        print(f"    Output:   {output2}")
        print(f"    Expected: {expected2}")

        if np.allclose(output2, expected2, atol=1e-5):
            print("\n    PASS: Second bake also matches!")
        else:
            print(f"\n    FAIL: Output mismatch! Max diff = {np.max(np.abs(output2 - expected2))}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

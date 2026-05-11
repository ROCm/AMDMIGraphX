"""
Generate a simple ONNX model with external weights and two weight directories
for testing the weight_params_example.

Creates:
  test_model/
    model.onnx          -- conv model referencing external "weights.bin"
    weights_v1/
      weights.bin        -- all weights = 1.0
    weights_v2/
      weights.bin        -- all weights = 2.0

Usage:
  python3 generate_test_model.py
  python3 weight_params_example.py test_model/model.onnx test_model/weights_v1 test_model/weights_v2
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto


def write_raw_weights(path, *arrays):
    """Write multiple numpy arrays concatenated into a single binary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for arr in arrays:
            f.write(arr.astype(np.float32).tobytes())


def build_model(out_dir):
    """Build ONNX model with external weight references."""
    os.makedirs(out_dir, exist_ok=True)

    weight_shape = [4, 1, 3, 3]
    bias_shape = [4]
    weight_nbytes = int(np.prod(weight_shape)) * 4
    bias_nbytes = int(np.prod(bias_shape)) * 4

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

    conv_weight = onnx.TensorProto()
    conv_weight.name = "conv.weight"
    conv_weight.data_type = TensorProto.FLOAT
    conv_weight.dims.extend(weight_shape)
    conv_weight.data_location = TensorProto.EXTERNAL
    conv_weight.external_data.add(key="location", value="weights.bin")
    conv_weight.external_data.add(key="offset", value="0")
    conv_weight.external_data.add(key="length", value=str(weight_nbytes))

    conv_bias = onnx.TensorProto()
    conv_bias.name = "conv.bias"
    conv_bias.data_type = TensorProto.FLOAT
    conv_bias.dims.extend(bias_shape)
    conv_bias.data_location = TensorProto.EXTERNAL
    conv_bias.external_data.add(key="location", value="weights.bin")
    conv_bias.external_data.add(key="offset", value=str(weight_nbytes))
    conv_bias.external_data.add(key="length", value=str(bias_nbytes))

    conv_node = helper.make_node("Conv", ["input", "conv.weight", "conv.bias"], ["output"],
                                 kernel_shape=[3, 3])

    graph = helper.make_graph([conv_node], "test_weight_params",
                              [X], [Y],
                              initializer=[conv_weight, conv_bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    model_path = os.path.join(out_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    print(f"  Model:   {model_path}")

    return weight_shape, bias_shape


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_model")

    print("Generating test model and weight files...")
    weight_shape, bias_shape = build_model(out_dir)

    weights_v1 = np.ones(weight_shape, dtype=np.float32)
    bias_v1 = np.zeros(bias_shape, dtype=np.float32)
    v1_dir = os.path.join(out_dir, "weights_v1")
    write_raw_weights(os.path.join(v1_dir, "weights.bin"), weights_v1, bias_v1)
    print(f"  Weights: {v1_dir}/weights.bin  (weights=1.0, bias=0.0)")

    weights_v2 = np.full(weight_shape, 2.0, dtype=np.float32)
    bias_v2 = np.ones(bias_shape, dtype=np.float32)
    v2_dir = os.path.join(out_dir, "weights_v2")
    write_raw_weights(os.path.join(v2_dir, "weights.bin"), weights_v2, bias_v2)
    print(f"  Weights: {v2_dir}/weights.bin  (weights=2.0, bias=1.0)")

    print()
    print("Run the example:")
    print(f"  python3 weight_params_example.py {out_dir}/model.onnx {v1_dir} {v2_dir}")


if __name__ == "__main__":
    main()

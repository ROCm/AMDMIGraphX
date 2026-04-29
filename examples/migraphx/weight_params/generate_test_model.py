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
import struct
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def write_raw_weights(path, *arrays):
    """Write multiple numpy arrays concatenated into a single binary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for arr in arrays:
            f.write(arr.astype(np.float32).tobytes())


def build_model_with_onnx(out_dir):
    """Build ONNX model using the onnx Python package."""
    os.makedirs(out_dir, exist_ok=True)

    weight_shape = [4, 1, 3, 3]
    bias_shape = [4]
    weight_size = int(np.prod(weight_shape))
    bias_size = int(np.prod(bias_shape))

    weight_nbytes = weight_size * 4
    bias_nbytes = bias_size * 4

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


def build_model_raw(out_dir):
    """Build a minimal ONNX protobuf by hand (no onnx package needed)."""
    os.makedirs(out_dir, exist_ok=True)

    weight_shape = [4, 1, 3, 3]
    bias_shape = [4]
    weight_nbytes = int(np.prod(weight_shape)) * 4
    bias_nbytes = int(np.prod(bias_shape)) * 4

    def encode_varint(value):
        result = bytearray()
        while value > 0x7F:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    def encode_field(field_num, wire_type, data):
        tag = encode_varint((field_num << 3) | wire_type)
        if wire_type == 2:
            return tag + encode_varint(len(data)) + data
        elif wire_type == 0:
            return tag + encode_varint(data)
        return tag + data

    def make_string_string_entry(key, value):
        inner = encode_field(1, 2, key.encode()) + encode_field(2, 2, value.encode())
        return inner

    def make_external_tensor(name, dtype, dims, location, offset, length):
        data = b""
        for d in dims:
            data += encode_field(1, 0, d)
        data += encode_field(2, 0, dtype)
        data += encode_field(4, 2, name.encode())
        for key, val in [("location", location), ("offset", str(offset)), ("length", str(length))]:
            entry = make_string_string_entry(key, val)
            data += encode_field(13, 2, entry)
        data += encode_field(14, 0, 1)  # data_location = EXTERNAL
        return data

    def make_value_info(name, elem_type, dims):
        tensor_type = encode_field(1, 0, elem_type)
        shape_data = b""
        for d in dims:
            dim = encode_field(1, 0, d)
            shape_data += encode_field(1, 2, dim)
        tensor_type += encode_field(2, 2, shape_data)
        type_proto = encode_field(1, 2, tensor_type)
        return encode_field(1, 2, name.encode()) + encode_field(2, 2, type_proto)

    def make_node(op_type, inputs, outputs, attrs=None):
        data = b""
        for i in inputs:
            data += encode_field(1, 2, i.encode())
        for o in outputs:
            data += encode_field(2, 2, o.encode())
        data += encode_field(4, 2, op_type.encode())
        if attrs:
            for attr in attrs:
                data += encode_field(5, 2, attr)
        return data

    def make_attr_ints(name, values):
        data = encode_field(1, 2, name.encode())
        for v in values:
            data += encode_field(7, 0, v)
        data += encode_field(20, 0, 7)  # type = INTS
        return data

    w_tensor = make_external_tensor("conv.weight", 1, weight_shape, "weights.bin", 0, weight_nbytes)
    b_tensor = make_external_tensor("conv.bias", 1, bias_shape, "weights.bin", weight_nbytes, bias_nbytes)

    input_vi = make_value_info("input", 1, [1, 1, 8, 8])
    output_vi = make_value_info("output", 1, [1, 4, 6, 6])

    kernel_attr = make_attr_ints("kernel_shape", [3, 3])
    conv_node = make_node("Conv", ["input", "conv.weight", "conv.bias"], ["output"], [kernel_attr])

    graph = b""
    graph += encode_field(1, 2, conv_node)
    graph += encode_field(2, 2, b"test_weight_params")
    graph += encode_field(5, 2, w_tensor)
    graph += encode_field(5, 2, b_tensor)
    graph += encode_field(11, 2, input_vi)
    graph += encode_field(12, 2, output_vi)

    opset = encode_field(2, 0, 13)

    model = b""
    model += encode_field(1, 0, 7)  # ir_version
    model += encode_field(7, 2, graph)
    model += encode_field(8, 2, opset)

    model_path = os.path.join(out_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(model)
    print(f"  Model:   {model_path}")

    return weight_shape, bias_shape


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_model")

    print("Generating test model and weight files...")

    if HAS_ONNX:
        weight_shape, bias_shape = build_model_with_onnx(out_dir)
    else:
        print("  (onnx package not found, using raw protobuf builder)")
        weight_shape, bias_shape = build_model_raw(out_dir)

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

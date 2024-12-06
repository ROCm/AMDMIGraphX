import migraphx
import onnx
from onnx import helper, TensorProto, checker
import numpy as np
import os


# Utility function to map MIGraphX types to ONNX data types
def get_dtype(instruction):
    type_mapping = {'float_type': TensorProto.FLOAT}
    return type_mapping[instruction.shape().type_string()]


# Utility function to get the shape of an instruction
def get_shape(instruction):
    if isinstance(instruction, list):
        raise ValueError("Expected instruction, got a list.")
    return instruction.shape().lens()


# Utility function to map MIGraphX operations to ONNX operations
def map_operation(operation):
    mxr_to_onnx_op = {
        "dot": "MatMul",
        "mul": "MatMul",
        "add": "Add",
        "multibroadcast": "Expand",
        "erf": "Erf",
        "tanh": "Tanh",
        "exp": "Exp",
        "div": "Div",
        "relu": "Relu"
    }

    if operation not in mxr_to_onnx_op:
        raise NotImplementedError(f"Operation '{operation}' is not supported.")
    return mxr_to_onnx_op[operation]


# Helper function to create ONNX nodes for specific operations
def create_node(instruction, parameters, node_name, n, initializers):
    if node_name == "multibroadcast" or node_name == "reshape":
        shape_key = "out_lens" if node_name == "multibroadcast" else "dims"
        shape_array = np.array(parameters[shape_key], dtype=np.int64)
        initializer_name = f"{node_name}_shape_{n}"

        initializers.append(
            helper.make_tensor(name=initializer_name,
                               data_type=TensorProto.INT64,
                               dims=shape_array.shape,
                               vals=shape_array.flatten().tolist()))
        return helper.make_node(
            map_operation(node_name),
            inputs=[str(hash(i))
                    for i in instruction.inputs()] + [initializer_name],
            outputs=[str(hash(instruction))])

    elif node_name == "transpose":
        return helper.make_node(
            "Transpose",
            inputs=[str(hash(i)) for i in instruction.inputs()],
            outputs=[str(hash(instruction))],
            perm=parameters["permutation"])

    elif node_name == "convolution":
        return helper.make_node(
            "Conv",
            inputs=[str(hash(i)) for i in instruction.inputs()],
            outputs=[str(hash(instruction))],
            dilations=parameters["dilation"],
            group=parameters["group"],
            pads=parameters["padding"],
            strides=parameters["stride"])

    return helper.make_node(
        map_operation(node_name),
        inputs=[str(hash(i)) for i in instruction.inputs()],
        outputs=[str(hash(instruction))])


# Main function to convert MIGraphX module to ONNX model
def generate_onnx(module):
    inputs = {}
    operations = []
    initializers = []
    n = 0  # Node counter
    output = None

    for instruction in module:
        op_name = instruction.op().name()

        # Handle input nodes
        if op_name in ["@literal", "@param"]:
            inputs[str(hash(instruction))] = helper.make_tensor_value_info(
                str(hash(instruction)), get_dtype(instruction),
                get_shape(instruction))

        # Handle computational nodes
        elif "@" not in op_name:
            n += 1
            parameters = instruction.op().values()
            operations.append(
                create_node(instruction, parameters, op_name, n, initializers))

        # Handle return node
        elif op_name == "@return":
            output = helper.make_tensor_value_info(str(hash(instruction)),
                                                   get_dtype(instruction),
                                                   get_shape(instruction))

    # Create the ONNX graph
    graph = helper.make_graph(nodes=operations,
                              name="Graph",
                              inputs=list(inputs.values()),
                              initializer=initializers,
                              outputs=[output] if output else [])

    return helper.make_model(graph, producer_name="onnx-dot-add-example")


# Main function to process MIGraphX files and generate ONNX models
def main(directory_path="mxr/"):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            program = migraphx.load(file_path)
            module = program.get_main_module()
            model = generate_onnx(module)

            # Validate the generated ONNX model
            try:
                checker.check_model(model)
                print(f"ONNX model for {file_name} is valid.")
            except onnx.checker.ValidationError as e:
                print(f"Validation failed for {file_name}: {e}")
            except Exception as e:
                print(
                    f"Unexpected error during validation for {file_name}: {e}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main()

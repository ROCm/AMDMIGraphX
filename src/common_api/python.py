import migraphx as mgx
import migraphx.common_api as trt


import os
import argparse

# # This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np

# import tensorrt as trt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], ".."))
# import common

class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# # You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# # The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1)
#     # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            print("Num errors: ", parser.num_errors)
            # for error in range(parser.num_errors):
            #     print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    print("Built serialized network")
    runtime = trt.Runtime(TRT_LOGGER)
    print("Created runtime")
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    print("Created runtime engine");
    return engine


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = (
            np.asarray(image.resize((w, h), Image.LANCZOS))
            .transpose([2, 0, 1])
            .astype(trt.nptype(ModelData.DTYPE))
            .ravel()
        )
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError(
                "Could not find {:}. Searched in data paths: {:}\n{:}".format(
                    filename, data_paths, err_msg
                )
            )
    return found_files

def find_sample_data(
    description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""
):
    """
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    """

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--datadir",
        help="Location of the TensorRT sample data directory, and any additional data directories.",
        action="append",
        default=[kDEFAULT_DATA_ROOT],
    )
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print(
                    "WARNING: "
                    + data_path
                    + " does not exist. Trying "
                    + data_dir
                    + " instead."
                )
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print(
                "WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(
                    data_path
                )
            )
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)

def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    _, data_files = find_sample_data(
        description="Runs a ResNet50 network with a TensorRT inference engine.",
        subfolder="resnet50",
        find_files=[
            "binoculars.jpeg",
            "reflex_camera.jpeg",
            "tabby_tiger_cat.jpg",
            ModelData.MODEL_PATH,
            "class_labels.txt",
        ],
    )
#     # Get test images, models and labels.
    test_images = data_files[0:3]
    onnx_model_file, labels_file = data_files[3:]
    labels = open(labels_file, "r").read().split("\n")

#     # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)
#     # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
#     # Allocate buffers and create a CUDA stream.
#     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
#     # Contexts are used to perform inference.
#     context = engine.create_execution_context()

#     # Load a normalized test case into the host input page-locked buffer.
#     test_image = random.choice(test_images)
#     test_case = load_normalized_test_case(test_image, inputs[0].host)
#     # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
#     # probability that the image corresponds to that label
#     trt_outputs = common.do_inference(
#         context,
#         engine=engine,
#         bindings=bindings,
#         inputs=inputs,
#         outputs=outputs,
#         stream=stream,
#     )
#     # We use the highest probability as our prediction. Its index corresponds to the predicted label.
#     pred = labels[np.argmax(trt_outputs[0])]
#     common.free_buffers(inputs, outputs, stream)
#     if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
#         print("Correctly recognized " + test_case + " as " + pred)
#     else:
#         print("Incorrectly recognized " + test_case + " as " + pred)


if __name__ == "__main__":
    main()

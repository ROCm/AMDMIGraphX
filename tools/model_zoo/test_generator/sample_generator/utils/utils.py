import onnx
import logging
import os
import requests


def get_model_io(model_path):
    model = onnx.load(model_path)
    outputs = [node.name for node in model.graph.output]

    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    inputs = list(set(input_all) - set(input_initializer))

    return inputs, outputs


def numpy_to_pb(name, np_data, out_filename):
    """Convert numpy data to a protobuf file."""

    tensor = onnx.numpy_helper.from_array(np_data, name)
    onnx.save_tensor(tensor, out_filename)


def download(url, filename, quiet=False):
    try:
        from tqdm import tqdm
    except:
        quiet = True

    logging.debug(f"Download {filename} from {url}")
    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    if not quiet:
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit="iB",
                            unit_scale=True)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            if not quiet:
                progress_bar.update(len(data))
            f.write(data)
    if not quiet:
        progress_bar.close()


def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.split("\n")

import numpy as np
from datasets import load_dataset

from preprocess import process_image

# TODO
# Class
#   dataset_iterator()
#   data_transform()

IMAGENET_VAL_DATASET_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"


def get_imagenet_dataset_iterator():
    print(f"Load dataset from {IMAGENET_VAL_DATASET_URL}")
    return load_dataset("webdataset",
                        data_files={"val": IMAGENET_VAL_DATASET_URL},
                        split="val",
                        streaming=True)


def imagenet_transform(inputs, data):
    assert len(inputs) == 1
    IMAGENET_MEANS = [123.68, 116.78, 103.94]  # RGB
    img_data = process_image(data["jpeg"], IMAGENET_MEANS)
    # add batch dimension
    img_data = img_data[np.newaxis, :]
    return {inputs[0]: img_data}

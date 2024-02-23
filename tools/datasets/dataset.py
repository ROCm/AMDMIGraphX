import abc
import numpy as np
from datasets import load_dataset

from preprocess import process_image


class BaseDataset(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def transform(self, inputs, data, prepocess_fn):
        pass

    @abc.abstractmethod
    def name(self):
        pass


class ImageNet2012Val(BaseDataset):

    def __init__(self):
        self.url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

    def __iter__(self):
        print(f"Load dataset from {self.url}")
        self.dataset = iter(
            load_dataset("webdataset",
                         data_files={"val": self.url},
                         split="val",
                         streaming=True))
        return self.dataset

    def __next__(self):
        return next(self.dataset)

    def transform(self, inputs, data, prepocess_fn):
        assert len(inputs) == 1
        img_data = prepocess_fn(data["jpeg"])
        assert (img_data.shape == (1, 3, 224, 224))
        return {inputs[0]: img_data}

    def name(self):
        return "imagenet-2012-val"

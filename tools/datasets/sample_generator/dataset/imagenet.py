from .base import BaseDataset
from datasets import load_dataset


class ImageNet2012Val(BaseDataset):
    @property
    def url(self):
        return "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

    @property
    def split(self):
        return "val"

    @property
    def name(self):
        return "imagenet-2012-val"

    def __iter__(self):
        print(f"Load dataset from {self.url}")
        self.dataset = iter(
            load_dataset("webdataset",
                         data_files={self.split: self.url},
                         split=self.split,
                         streaming=True))
        return self.dataset

    def __next__(self):
        return next(self.dataset)

    def transform(self, inputs, data, prepocess_fn):
        assert len(inputs) == 1
        img_data = prepocess_fn(data["jpeg"])
        assert (img_data.shape == (1, 3, 224, 224))
        return {inputs[0]: img_data}

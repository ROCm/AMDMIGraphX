import abc
from utils import download
from preprocess import process_image
import PIL
from optimum.exporters.onnx import main_export
from transformers import AutoImageProcessor


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def download(self, output_folder):
        pass

    @abc.abstractmethod
    def preprocess(self, image_data):
        pass

    @abc.abstractmethod
    def name(self):
        pass


class ResNet50_v1(BaseModel):

    def __init__(self):
        self.url = f"https://zenodo.org/record/2592612/files/resnet50_v1.onnx"

    def download(self, output_folder):
        filepath = f"{output_folder}/model.onnx"
        print(f"Download model from {self.url} to {filepath}")
        download(self.url, filepath)
        return filepath

    def preprocess(self, image_data):
        IMAGENET_MEANS = [123.68, 116.78, 103.94]  # RGB
        return process_image(image_data, means=IMAGENET_MEANS)

    def name(self):
        return "resnet50_v1"


class ResNet50_v1_5(BaseModel):

    def __init__(self):
        self.model_id = "microsoft/resnet-50"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    def download(self, output_folder):
        main_export(self.model_id, output=output_folder)
        return f"{output_folder}/model.onnx"

    def preprocess(self, image_data):
        return self.processor(image_data, return_tensors="np")['pixel_values']

    def name(self):
        return "resnet50_v1.5"

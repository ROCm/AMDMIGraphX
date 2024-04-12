import abc
import os
from optimum.exporters.onnx import main_export
from ..utils import download


class BaseModel(abc.ABC):
    @property
    @abc.abstractmethod
    def model_id(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @classmethod
    @abc.abstractmethod
    def download(self, output_folder):
        pass

    @classmethod
    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @classmethod
    def is_decoder(self):
        return False


class DecoderModel(BaseModel):
    def is_decoder(self):
        return True

    @classmethod
    @abc.abstractmethod
    def decode_step(self, *args, **kwargs):
        pass


class SingleModelDownloadMixin(object):
    def download(self, output_folder, force=False):
        filepath = f"{output_folder}/model.onnx"
        if force or not os.path.isfile(filepath):
            print(f"Download model from {self.model_id} to {filepath}")
            download(self.model_id, filepath)
        return filepath


class SingleOptimumHFModelDownloadMixin(object):
    def download(self, output_folder, force=False):
        filepath = f"{output_folder}/model.onnx"
        if force or not os.path.isfile(filepath):
            print(f"Download model from {self.model_id} to {filepath}")
            main_export(self.model_id, output=output_folder)
        return filepath


class EncoderDecoderOptimumHFModelDownloadMixin(object):
    def download(self, output_folder, force=False):
        filepath = f"{output_folder}/model.onnx"
        if force or not os.path.isfile(filepath):
            print(f"Download model from {self.model_id} to {filepath}")
            # monolith forces export into one model
            main_export(self.model_id, output=output_folder, monolith=True)
        return filepath

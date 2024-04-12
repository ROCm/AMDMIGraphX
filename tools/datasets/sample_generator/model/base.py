import abc
from optimum.exporters.onnx import main_export
from ..utils import download


class BaseModel(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def __init__(self):
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
    @abc.abstractmethod
    def name(self):
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


class SingleURLDownloadMixin(object):

    def download(self, output_folder):
        filepath = f"{output_folder}/model.onnx"
        print(f"Download model from {self.url} to {filepath}")
        download(self.url, filepath)
        return filepath


class SingleOptimumHFModelDownloadMixin(object):

    def download(self, output_folder):
        main_export(self.model_id, output=output_folder)
        return f"{output_folder}/model.onnx"


class EncoderDecoderOptimumHFModelDownloadMixin(object):

    def download(self, output_folder):
        # monolith forces export into one model
        main_export(self.model_id, output=output_folder, monolith=True)
        return f"{output_folder}/model.onnx"

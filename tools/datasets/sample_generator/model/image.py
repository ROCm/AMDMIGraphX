from .base import *
from .preprocess import process_image

from transformers import AutoImageProcessor
import timm


class AutoImageProcessorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self, image_data):
        return self.processor(image_data, return_tensors="np")


class ResNet50_v1(SingleModelDownloadMixin, BaseModel):
    @property
    def model_id(self):
        return f"https://zenodo.org/record/2592612/files/resnet50_v1.onnx"

    @property
    def name(self):
        return "resnet50_v1"

    def preprocess(self, image_data):
        IMAGENET_MEANS = [123.68, 116.78, 103.94]  # RGB
        return {
            "input_tensor:0": process_image(image_data, means=IMAGENET_MEANS)
        }


class ResNet50_v1_5(SingleOptimumHFModelDownloadMixin,
                    AutoImageProcessorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "microsoft/resnet-50"

    @property
    def name(self):
        return "resnet50_v1.5"


class VitBasePatch16_224(SingleOptimumHFModelDownloadMixin,
                         AutoImageProcessorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "google/vit-base-patch16-224"

    @property
    def name(self):
        return "vit-base-patch16-224"


class TIMM_MobileNetv3_large(SingleOptimumHFModelDownloadMixin, BaseModel):
    def __init__(self):
        data_config = timm.data.resolve_model_data_config(
            timm.create_model(self.model_id, pretrained=True))
        self.processor = timm.data.create_transform(**data_config,
                                                    is_training=False)

    @property
    def model_id(self):
        return "timm/mobilenetv3_large_100.ra_in1k"

    @property
    def name(self):
        return "timm-mobilenetv3-large"

    def preprocess(self, image_data):
        return {
            "pixel_values":
            self.processor(image_data).unsqueeze(0).cpu().detach().numpy()
        }


# TODO enable it when BiT is supported by optimum
class Bit50(SingleOptimumHFModelDownloadMixin, AutoImageProcessorHFMixin,
            BaseModel):
    @property
    def model_id(self):
        return "google/bit-50"

    @property
    def name(self):
        return "bit-50"

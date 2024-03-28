import abc
from utils import download
from preprocess import process_image
from optimum.exporters.onnx import main_export
import timm
from transformers import AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor
import numpy as np

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


class SingleOptimumHFModelDownloadMixin(object):

    def download(self, output_folder):
        main_export(self.model_id, output=output_folder)
        return f"{output_folder}/model.onnx"


class EncoderDecoderOptimumHFModelDownloadMixin(object):

    def download(self, output_folder):
        # monolith forces export into one model
        main_export(self.model_id, output=output_folder, monolith=True)
        return f"{output_folder}/model.onnx"


class AutoImageProcessorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self, image_data):
        return self.processor(image_data, return_tensors="np")['pixel_values']


class AutoTokenizerHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoTokenizer.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self, question, context, max_length):
        return self.processor(question,
                              context,
                              padding='max_length',
                              max_length=max_length,
                              truncation='only_second',
                              return_tensors="np")


class AutoFeatureExtractorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoFeatureExtractor.from_pretrained(
                self.model_id)
        return self._processor

    def preprocess(self, audio_data, sampling_rate):
        return self.processor(audio_data,
                              sampling_rate=sampling_rate,
                              return_tensors="np")


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


class ResNet50_v1_5(SingleOptimumHFModelDownloadMixin,
                    AutoImageProcessorHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "microsoft/resnet-50"

    def name(self):
        return "resnet50_v1.5"


class VitBasePatch16_224(SingleOptimumHFModelDownloadMixin,
                         AutoImageProcessorHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "google/vit-base-patch16-224"

    def name(self):
        return "vit-base-patch16-224"


# TODO enable it when BiT is supported by optimum
class Bit50(SingleOptimumHFModelDownloadMixin, AutoImageProcessorHFMixin,
            BaseModel):

    def __init__(self):
        self.model_id = "google/bit-50"

    def name(self):
        return "bit-50"


class TIMM_MobileNetv3_large(SingleOptimumHFModelDownloadMixin, BaseModel):

    def __init__(self):
        self.model_id = "timm/mobilenetv3_large_100.ra_in1k"
        data_config = timm.data.resolve_model_data_config(
            timm.create_model(self.model_id, pretrained=True))
        self.processor = timm.data.create_transform(**data_config,
                                                    is_training=False)

    def preprocess(self, image_data):
        return self.processor(image_data).unsqueeze(0).cpu().detach().numpy()

    def name(self):
        return "timm-mobilenetv3-large"


class DistilBERT_base_cased_distilled_SQuAD(SingleOptimumHFModelDownloadMixin,
                                            AutoTokenizerHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "distilbert/distilbert-base-cased-distilled-squad"

    def name(self):
        return "distilbert-base-cased-distilled-squad"


class RobertaBaseSquad2(SingleOptimumHFModelDownloadMixin,
                        AutoTokenizerHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "deepset/roberta-base-squad2"

    def name(self):
        return "roberta-base-squad2"


class Wav2Vec2_base_960h(SingleOptimumHFModelDownloadMixin,
                         AutoFeatureExtractorHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "facebook/wav2vec2-base-960h"

    def name(self):
        return "wav2vec2-base-960h"


# TODO enable it when encoder-decoder models work
class WhisperSmallEn(EncoderDecoderOptimumHFModelDownloadMixin,
                     AutoFeatureExtractorHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "openai/whisper-small.en"

    def preprocess(self, *args, **kwargs):
        # result only contains encoder data, extend it with decoder
        result = super().preprocess(*args, **kwargs)
        decoder_start_token_id = 50257  # <|startoftranscript|>
        eos_token_id = 50256  # "<|endoftext|>"
        notimestamps = 50362  # <|notimestamps|>
        sot = [decoder_start_token_id, notimestamps]
        max_length = 448
        result["decoder_input_ids"] = np.array(
            [sot + [eos_token_id] * (max_length - len(sot))])
        return result

    def name(self):
        return "whisper-small-en"


class GPTJ(SingleOptimumHFModelDownloadMixin, AutoTokenizerHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "EleutherAI/gpt-j-6b"

    def name(self):
        return "gpt-j"

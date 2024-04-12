import abc
from optimum.exporters.onnx import main_export
import timm
from transformers import AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor
import numpy as np

from ..utils import download
from .preprocess import process_image


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


class WhisperSmallEn(EncoderDecoderOptimumHFModelDownloadMixin,
                     AutoFeatureExtractorHFMixin, DecoderModel):

    def __init__(self):
        self.model_id = "openai/whisper-small.en"

        # Whisper specific part
        # TODO get these from config
        self.eos_token_id = 50256  # "<|endoftext|>"
        decoder_start_token_id = 50257  # <|startoftranscript|>
        notimestamps = 50362  # <|notimestamps|>
        sot = [decoder_start_token_id, notimestamps]
        max_length = 448
        self.initial_input_ids = np.array(
            [sot + [self.eos_token_id] * (max_length - len(sot))])

    def preprocess(self, *args, **kwargs):
        # result only contains encoder data, extend it with decoder
        result = super().preprocess(*args, **kwargs)
        result["decoder_input_ids"] = self.initial_input_ids
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["decoder_input_ids"][0] == self.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["decoder_input_ids"][0][timestep] = new_token
        return new_token == self.eos_token_id

    def name(self):
        return "whisper-small-en"


class GPTJ(SingleOptimumHFModelDownloadMixin, AutoTokenizerHFMixin, BaseModel):

    def __init__(self):
        self.model_id = "EleutherAI/gpt-j-6b"

    def name(self):
        return "gpt-j"

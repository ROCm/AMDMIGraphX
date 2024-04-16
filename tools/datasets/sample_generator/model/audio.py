from .base import *
import numpy as np
from transformers import AutoFeatureExtractor


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


class Wav2Vec2_base_960h(SingleOptimumHFModelDownloadMixin,
                         AutoFeatureExtractorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "facebook/wav2vec2-base-960h"

    @property
    def name(self):
        return "wav2vec2-base-960h"


class WhisperSmallEn(EncoderDecoderOptimumHFModelDownloadMixin,
                     AutoFeatureExtractorHFMixin, DecoderModel):
    def __init__(self):
        # Whisper specific part
        # TODO get these from config
        self.eos_token_id = 50256  # "<|endoftext|>"
        decoder_start_token_id = 50257  # <|startoftranscript|>
        notimestamps = 50362  # <|notimestamps|>
        sot = [decoder_start_token_id, notimestamps]
        max_length = 448
        self.initial_input_ids = np.array(
            [sot + [self.eos_token_id] * (max_length - len(sot))])

    @property
    def model_id(self):
        return "openai/whisper-small.en"

    @property
    def name(self):
        return "whisper-small-en"

    def preprocess(self, *args, **kwargs):
        # result only contains encoder data, extend it with decoder
        result = super().preprocess(*args, **kwargs)
        result["decoder_input_ids"] = np.copy(self.initial_input_ids)
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["decoder_input_ids"][0] == self.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["decoder_input_ids"][0][timestep] = new_token
        return new_token == self.eos_token_id

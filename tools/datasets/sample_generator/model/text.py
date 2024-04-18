from .base import *
import numpy as np
from transformers import AutoTokenizer


class AutoTokenizerHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoTokenizer.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self,
                   question,
                   context,
                   max_length,
                   truncation='only_second'):
        return self.processor(question,
                              context,
                              padding='max_length',
                              max_length=max_length,
                              truncation=truncation,
                              return_tensors="np")


class TextGenerationDecoderOnlyMixin(object):
    def __init__(self):
        # no pad token by default
        self.processor.pad_token = self.processor.eos_token

    def preprocess(self, *args, **kwargs):
        # swap squad's default "question - answer" order
        new_args, new_kwargs = list(args), kwargs
        new_args[0], new_args[1] = new_args[1], new_args[0]
        new_kwargs["truncation"] = "only_first"
        result = super().preprocess(*new_args, **new_kwargs)

        # result only contains "input_ids" and "attention_mask", extend it with "position_ids"
        result["position_ids"] = np.arange(0,
                                           len(result["input_ids"][0]),
                                           dtype=np.int64)
        result["position_ids"] = result["position_ids"][np.newaxis]
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["input_ids"][0] == self.processor.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["input_ids"][0][timestep] = new_token
        input_map["attention_mask"][0][timestep] = 1
        return new_token == self.processor.eos_token_id


class DistilBERT_base_cased_distilled_SQuAD(SingleOptimumHFModelDownloadMixin,
                                            AutoTokenizerHFMixin, BaseModel):
    @property
    def model_id(self):
        return "distilbert/distilbert-base-cased-distilled-squad"

    @property
    def name(self):
        return "distilbert-base-cased-distilled-squad"


class RobertaBaseSquad2(SingleOptimumHFModelDownloadMixin,
                        AutoTokenizerHFMixin, BaseModel):
    @property
    def model_id(self):
        return "deepset/roberta-base-squad2"

    @property
    def name(self):
        return "roberta-base-squad2"


# Note: the inheritance order here important
class GPTJ(SingleOptimumHFModelDownloadMixin, TextGenerationDecoderOnlyMixin,
           AutoTokenizerHFMixin, DecoderModel):
    @property
    def model_id(self):
        return "EleutherAI/gpt-j-6b"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @property
    def name(self):
        return "gpt-j"


# Note: the inheritance order here important
class Llama2_7b_chat_hf(SingleOptimumHFModelDownloadMixin,
                        TextGenerationDecoderOnlyMixin, AutoTokenizerHFMixin,
                        DecoderModel):
    @property
    def model_id(self):
        return "meta-llama/Llama-2-7b-chat-hf"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @property
    def name(self):
        return "llama2-7b-chat-hf"


class T5_base(EncoderDecoderOptimumHFModelDownloadMixin, AutoTokenizerHFMixin,
              DecoderModel):
    def __init__(self):
        max_length = 384  # default for squad
        # Note: no real start token, it requires the pad token
        self.initial_input_ids = np.array(
            [[self.processor.pad_token_id] + [self.processor.eos_token_id] *
             (max_length - 1)])

    @property
    def model_id(self):
        return "google-t5/t5-base"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text2text-generation"

    @property
    def name(self):
        return "t5-base"

    def preprocess(self, *args, **kwargs):
        # only use squad's question
        new_args = ["translate English to French:", args[0]]
        result = super().preprocess(*new_args, **kwargs)
        result["decoder_input_ids"] = np.copy(self.initial_input_ids)
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["decoder_input_ids"][0] == self.processor.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["decoder_input_ids"][0][timestep] = new_token
        input_map["attention_mask"][0][timestep] = 1
        return new_token == self.processor.eos_token_id


class Gemma_2b_it(SingleOptimumHFModelDownloadMixin, AutoTokenizerHFMixin,
                  DecoderModel):
    @property
    def model_id(self):
        return "google/gemma-2b-it"

    @property
    def task(self):
        # override to ignore "with-past"
        return "text-generation"

    @property
    def name(self):
        return "gemma-2b-it"

    def preprocess(self, *args, **kwargs):
        # swap squad's default "question - answer" order
        new_args, new_kwargs = list(args), kwargs
        new_args[0], new_args[1] = new_args[1], new_args[0]
        new_kwargs["truncation"] = "only_first"
        result = super().preprocess(*new_args, **new_kwargs)
        # Note: padding-side is left, pos ids will be 1,1,1,...,0,1,2,3,...
        result["position_ids"] = np.cumsum(result["attention_mask"][0], -1) - 1
        result["position_ids"][:np.argmax(result["position_ids"] == 0)] = 1
        result["position_ids"] = result["position_ids"][np.newaxis]
        return result

    def decode_step(self, input_map, output_map):
        # The result is in the last logits
        new_token = np.argmax(output_map["logits"][0][-1])
        # Move everything left 1 step
        input_map["input_ids"][0] = np.roll(input_map["input_ids"][0], -1)
        input_map["input_ids"][0][-1] = new_token
        input_map["attention_mask"][0] = np.roll(
            input_map["attention_mask"][0], -1)
        input_map["attention_mask"][0][-1] = 1
        input_map["position_ids"][0] = np.roll(input_map["position_ids"][0],
                                               -1)
        input_map["position_ids"][0][-1] += input_map["position_ids"][0][-2]
        return new_token == self.processor.eos_token_id

from .base import *

from transformers import AutoTokenizer


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


class GPTJ(SingleOptimumHFModelDownloadMixin, AutoTokenizerHFMixin, BaseModel):
    @property
    def model_id(self):
        return "EleutherAI/gpt-j-6b"

    @property
    def name(self):
        return "gpt-j"

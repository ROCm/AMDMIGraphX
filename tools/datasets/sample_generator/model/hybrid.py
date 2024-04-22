from .base import *
from ..utils import get_imagenet_classes
from transformers import AutoProcessor


class AutoProcessorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self, images, text, max_length=384):
        return self.processor(images=images,
                              text=text,
                              max_length=max_length,
                              padding='max_length',
                              return_tensors="np")


class ClipVitLargePatch14(OptimumHFModelDownloadMixin, AutoProcessorHFMixin,
                          BaseModel):
    def __init__(self):
        import random
        random.seed(42)
        # cache a few labels, the full 1000 label is overkill
        self.imagenet_labels = random.sample(get_imagenet_classes(), 10)

    @property
    def model_id(self):
        return "openai/clip-vit-large-patch14"

    @staticmethod
    def name():
        return "clip-vit-large-patch14"

    def preprocess(self, *args, **kwargs):
        # extend image with imagenet labels
        new_args, new_kwargs = list(args), kwargs
        new_args.append(self.imagenet_labels)
        new_kwargs["max_length"] = 77
        result = super().preprocess(*new_args, **new_kwargs)
        return result

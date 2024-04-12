from .base import BaseDataset, ValidationDatasetHFIteratorMixin

class LibriSpeechASR(ValidationDatasetHFIteratorMixin, BaseDataset):

    def __init__(self):
        self.url = "librispeech_asr"
        # override default split
        self.split = "validation.clean"

    def transform(self, inputs, data, prepocess_fn):
        result = prepocess_fn(data["audio"]["array"],
                              data["audio"]["sampling_rate"])
        inputs, keys = sorted(inputs), sorted(list(result.keys()))
        assert inputs == keys, f"{inputs = } == {keys = }"
        # The result should be a simple dict, the preproc returns a wrapped class, dict() will remove it
        return dict(result)

    def name(self):
        return "librispeech-asr"

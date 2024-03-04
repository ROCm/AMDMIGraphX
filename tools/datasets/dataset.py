import abc
from datasets import load_dataset


class BaseDataset(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def __init__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def __iter__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def __next__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def name(self):
        pass


class ValidationDatasetHFIteratorMixin(object):

    split = "validation"

    def __iter__(self):
        print(f"Load dataset from {self.url} using {self.split} split")
        self.dataset = iter(
            load_dataset(self.url, split=self.split, streaming=True))
        return self.dataset

    def __next__(self):
        return next(self.dataset)


class ImageNet2012Val(BaseDataset):

    def __init__(self):
        self.url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

    def __iter__(self):
        print(f"Load dataset from {self.url}")
        self.dataset = iter(
            load_dataset("webdataset",
                         data_files={"val": self.url},
                         split="val",
                         streaming=True))
        return self.dataset

    def __next__(self):
        return next(self.dataset)

    def transform(self, inputs, data, prepocess_fn):
        assert len(inputs) == 1
        img_data = prepocess_fn(data["jpeg"])
        assert (img_data.shape == (1, 3, 224, 224))
        return {inputs[0]: img_data}

    def name(self):
        return "imagenet-2012-val"


class SQuADTransformMixin(object):

    def transform(self, inputs, data, prepocess_fn):
        result = prepocess_fn(data["question"],
                              data["context"],
                              max_length=384)
        inputs, keys = sorted(inputs), sorted(list(result.keys()))
        assert inputs == keys, f"{inputs = } == {keys = }"
        # The result should be a simple dict, the preproc returns a wrapped class, dict() will remove it
        return dict(result)


class SQuADv1_1(SQuADTransformMixin, BaseDataset):

    def __init__(self):
        self.url = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"

    def __iter__(self):
        print(f"Load dataset from {self.url}")
        self.dataset = ({
            "context": paragraph["context"],
            "question": qas["question"]
        } for data in load_dataset("json",
                                   data_files={"val": self.url},
                                   split="val",
                                   field="data",
                                   streaming=True)
                        for paragraph in data["paragraphs"]
                        for qas in paragraph["qas"])
        return self.dataset

    def __next__(self):
        return next(self.dataset)

    def name(self):
        return "squad-v1.1"


class SQuAD_HF(ValidationDatasetHFIteratorMixin, SQuADTransformMixin,
               BaseDataset):

    def __init__(self):
        self.url = "squad"

    def name(self):
        return "squad-hf"


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

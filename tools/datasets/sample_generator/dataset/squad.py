from .base import BaseDataset, ValidationDatasetHFIteratorMixin
from datasets import load_dataset

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

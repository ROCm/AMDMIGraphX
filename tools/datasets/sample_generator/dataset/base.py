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

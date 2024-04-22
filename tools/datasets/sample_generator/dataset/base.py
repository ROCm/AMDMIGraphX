import abc
from datasets import load_dataset


class BaseDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def url(self):
        pass

    @property
    @abc.abstractmethod
    def split(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        pass


class ValidationDatasetHFIteratorMixin(object):
    def __iter__(self):
        print(f"Load dataset from {self.url} using {self.split} split")
        self.dataset = iter(
            load_dataset(self.url, split=self.split, streaming=True))
        return self.dataset

    def __next__(self):
        return next(self.dataset)

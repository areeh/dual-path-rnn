from abc import ABCMeta
from abc import abstractmethod
from typing import Iterator


class Data(Metaclass=ABCMeta):
    @abstractmethod
    def batch_generator(self, batch_size: int, num_batches: int) -> Iterator:
        pass

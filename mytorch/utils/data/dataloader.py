from typing import Generic, TypeVar, Tuple, Optional
from mytorch._tensor import Tensor
import numpy as np


T_co = TypeVar('T_co', covariant=True)


class DataLoader(Generic[T_co]):
    def __init__(self, dataset: Generic[T_co], batch_size: Optional[int] = 1, shuffle: Optional[bool] = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.start = 0
        return self

    def __next__(self):
        if self.start >= len(self.indices):
            raise StopIteration
        end = self.start + self.batch_size
        batch_indices = self.indices[self.start:end]
        batch = [self.dataset[i] for i in batch_indices]

        # 自动拆包为 (x_batch, y_batch, ...)
        if isinstance(batch[0], (list, tuple)):
            batch = tuple(np.stack(samples) for samples in zip(*batch))
        else:
            batch = np.stack(batch)

        self.start = end
        return batch



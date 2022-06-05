"""

Inspired by https://github.com/robtandy/randomdict/blob/master/randomdict.py
"""
from collections.abc import MutableMapping

import numpy as np
from typing import Union


class CacheDict(MutableMapping):
    """Dictionary that tracks the sum of its non-NaN values and its non-NaN size."""

    def __init__(self):
        self.dict = {}
        self.sum = 0.0
        self.size = 0

    def __setitem__(self, key: Union[str, float, int], value: np.ndarray):
        if key in self.dict:
            raise NotImplementedError

        self.sum += np.nansum(value)
        self.size += np.count_nonzero(~np.isnan(value))

        self.dict[key] = value

    def __delitem__(self, key):
        raise NotImplementedError

    def __getitem__(self, key: Union[str, float, int]) -> np.ndarray:
        return self.dict[key]

    def __iter__(self):
        return iter(self.dict)

    def __len__(self) -> int:
        return len(self.dict)

    @property
    def mean(self) -> float:
        return self.sum / self.size

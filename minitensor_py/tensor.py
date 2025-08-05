import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

import minitensor as mt

class Tensor:
    def __init__(self, data: List, shape: List[int]):
        if not data:
            raise ValueError("ERROR: Data cannot be empty.")
        elif not shape:
            raise ValueError("ERROR: Shape cannot be empty.")

        first_el = data[0]
        if isinstance(first_el, float):
            self._tensor = mt.float32.Tensor(data, shape)
            self.dtype = "float32"
        elif isinstance(first_el, int):
            self._tensor = mt.int32.Tensor(data, shape)
            self.dtype = "int32"
        else:
            raise TypeError("ERROR: Unsupported data type. Only int, float and double are supported.")

    @property
    def shape(self) -> tuple:
        return tuple(self._tensor.shape)

    @property
    def size(self) -> int:
        return self._tensor.size
    
    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    def to_list(self) -> list:
        return self._tensor.to_list()

    def __repr__(self):
        return repr(self._tensor)
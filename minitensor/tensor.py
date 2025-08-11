import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__)))

from . import minitensor_cpp as mtc

class Tensor:
    def __init__(self, data: List, shape: List[int], dtype: str):
        if not data:
            raise ValueError("ERROR: Data cannot be empty.")
        elif not shape:
            raise ValueError("ERROR: Shape cannot be empty.")

        first_el = data[0]
        if (not isinstance(first_el, int)) and (not isinstance(first_el, float)):
            raise TypeError("ERROR: Unsupported data type. Only int, float and double are supported.")

        self.dtype = dtype
        if (dtype == "int32") or (dtype == "int"):
            self._tensor = mtc.int32.Tensor(data, shape)

        elif (dtype == "float32") or (dtype == "float"):
            self._tensor = mtc.float32.Tensor(data, shape)

        elif (dtype == "float64") or (dtype == "double"):
            self._tensor = mtc.float64.Tensor(data, shape)

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
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor + other._tensor

        elif (isinstance(other, int)) or (isinstance(other, float)):
            result = self._tensor + other

        else:
            raise TypeError(f"Unsupported operand type for +: 'Tensor' and '{type(other).__name__}'")

        new_python_tensor = Tensor.__new__(Tensor)
        new_python_tensor._tensor = result
        new_python_tensor.dtype = self.dtype
        return new_python_tensor

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor - other._tensor

        elif (isinstance(other, int)) or (isinstance(other, float)):
            result = self._tensor - other

        else:
            raise TypeError(f"Unsupported operand type for -: 'Tensor' and '{type(other).__name__}'")
        
        new_python_tensor = Tensor.__new__(Tensor)
        new_python_tensor._tensor = result
        new_python_tensor.dtype = self.dtype
        return new_python_tensor

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor * other._tensor

        elif (isinstance(other, int)) or (isinstance(other, float)):
            result = self._tensor * other

        else:
            raise TypeError(f"Unsupported operand type for *: 'Tensor' and '{type(other).__name__}'")

        new_python_tensor = Tensor.__new__(Tensor)
        new_python_tensor._tensor = result
        new_python_tensor.dtype = self.dtype
        return new_python_tensor

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor / other._tensor

        elif (isinstance(other, int)) or (isinstance(other, float)):
            result = self._tensor / other

        else:
            raise TypeError(f"Unsupported operand type for /: 'Tensor' and '{type(other).__name__}'")
        
        new_python_tensor = Tensor.__new__(Tensor)
        new_python_tensor._tensor = result
        new_python_tensor.dtype = self.dtype
        return new_python_tensor

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

def tensor(data: list, shape: list = None, dtype: str = None) -> Tensor:
    if not data:
        raise ValueError("ERROR: Data cannot be empty.")

    if dtype is None:
        first_el = data[0]
        if isinstance(first_el, int):
            dtype = "int32"
        elif isinstance(first_el, float):
            dtype = "float32"
        else:
            raise TypeError("ERROR: Unsupported data type. Only int, float and double are supported.")

    if shape is None:
        shape = [len(data)]

    if (dtype == "int32") or (dtype == "int"):
        return Tensor(data, shape, "int")

    elif (dtype == "float32") or (dtype == "float"):
        return Tensor(data, shape, "float")

    elif (dtype == "float64") or (dtype == "double"):
        return Tensor(data, shape, "double")

    else:
        raise TypeError("ERROR: Unsupported data type. Only int, float and double are supported.")
from typing import List
import math

from minitensor.backend import get_backend

class Tensor:
    def __init__(self, data: List, shape: List[int], dtype: str, requires_grad: bool = False):
        if not data:
            raise ValueError("ERROR: Data cannot be empty.")
        elif not shape:
            raise ValueError("ERROR: Shape cannot be empty.")

        first_el = data[0]
        if not isinstance(first_el, (int, float)):
            raise TypeError("ERROR: Unsupported data type. Only int, float and double are supported.")

        self.dtype = dtype

        self.backend = get_backend(dtype)

        self._tensor = self.backend.Tensor(data, shape, requires_grad)

    @property
    def shape(self) -> tuple:
        return tuple(self._tensor.shape)

    @property
    def size(self) -> int:
        return self._tensor.size

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def stride(self) -> tuple:
        return self._tensor.stride

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self._tensor.requires_grad = value

    @property
    def grad(self):
        if self.requires_grad and self._tensor.grad:
            requires_grad = False 
            return self._new_tensor(self._tensor.grad, self.dtype, requires_grad)
        return None

    def backward(self):
        self._tensor.backward()

    def zero_grad(self):
        self._tensor.zero_grad()

    @property
    def listed(self) -> list:
        return self._tensor.to_vector()

    @property
    def nested(self) -> list:
        return self._tensor.to_nested()

    @classmethod
    def _new_tensor(cls, result, dtype: str, requires_grad: bool=False):
        new_python_tensor = cls.__new__(cls)
        new_python_tensor._tensor = result
        new_python_tensor.dtype = dtype
        if requires_grad:
            new_python_tensor.requires_grad = True
        return new_python_tensor

    def __repr__(self):
        return repr(self._tensor)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor @ other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for @: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor + other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor + other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for +: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor - other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor - other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for -: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor * other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor * other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for *: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor / other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor / other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for /: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            result = other._tensor - self._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = other - self._tensor
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for -: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            result = other._tensor / self._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = other / self._tensor
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for /: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, self.dtype, requires_grad)

def _calculate_shape(data):
    if not isinstance(data, list):
        return []
    shape = [len(data)]
    if len(data) > 0:
        sub_shape = _calculate_shape(data[0])
        for item in data:
            if _calculate_shape(item) != sub_shape:
                raise ValueError("ERROR: All elements in a dimension must have the same shape.")
        shape.extend(sub_shape)
    return shape

def _flatten_nested_list(data):
    if not isinstance(data, list):
        return [data]
    flat_list = list()
    for item in data:
        flat_list.extend(_flatten_nested_list(item))
    return flat_list

def tensor(data, shape=None, dtype=None, requires_grad=False) -> Tensor:
    if not isinstance(data, list) or not data:
        raise ValueError("ERROR: Data must be a non-empty list.")

    calculated_shape = []
    flat_data = []

    if shape is not None:
        if any(isinstance(i, list) for i in data):
             raise ValueError("ERROR: Cannot provide an explicit shape for a nested list.")

        expected_elements = math.prod(shape)
        if len(data) != expected_elements:
            raise ValueError(f"ERROR: Shape {shape} requires {expected_elements} elements, but got {len(data)}.")

        flat_data = data
        calculated_shape = shape
    else:
        calculated_shape = _calculate_shape(data)
        flat_data = _flatten_nested_list(data)

    if dtype is None:
        first_el = flat_data[0]
        if isinstance(first_el, int):
            dtype = "int32"
        elif isinstance(first_el, float):
            dtype = "float32"
        else:
            raise TypeError("ERROR: Unsupported data type. Only int and float are supported.")

    return Tensor(flat_data, calculated_shape, dtype, requires_grad)
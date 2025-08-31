from typing import List

from minitensor.backend import get_backend, is_dtype_valid

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
            new_tensor = Tensor.__new__(Tensor)
            new_tensor._tensor = self._tensor.grad
            new_tensor.dtype = self.dtype
            return new_tensor
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

    def _new_tensor(self, result, requires_grad: bool):
        new_python_tensor = Tensor.__new__(Tensor)
        new_python_tensor._tensor = result
        new_python_tensor.dtype = self.dtype
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
        return self._new_tensor(result, requires_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor + other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor + other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for +: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor - other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor - other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for -: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor * other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor * other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for *: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, requires_grad)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = self._tensor / other._tensor
            requires_grad = self.requires_grad or other.requires_grad
        elif isinstance(other, (int, float)):
            result = self._tensor / other
            requires_grad = self.requires_grad
        else:
            raise TypeError(f"ERROR: Unsupported operand type for /: 'Tensor' and '{type(other).__name__}'")
        return self._new_tensor(result, requires_grad)

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
        return self._new_tensor(result, requires_grad)

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
        return self._new_tensor(result, requires_grad)

def tensor(data: list, shape: list = None, dtype: str = None, requires_grad: bool = False) -> Tensor:
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

    if is_dtype_valid(dtype):
        return Tensor(data, shape, dtype, requires_grad)
from minitensor.model import Module
from minitensor import Tensor
from minitensor.backend import get_backend

class Softmax(Module):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        backend = get_backend(x.dtype)

        result = backend.softmax(x._tensor, self.axis)

        return Tensor._new_tensor(result, x.dtype, x.requires_grad)

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"
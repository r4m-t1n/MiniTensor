from minitensor.model import Module
from minitensor import Tensor
from minitensor.backend import get_backend

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:

        backend = get_backend(x.dtype)

        result = backend.tanh(x._tensor)

        return Tensor._new_tensor(result, x.dtype, x.requires_grad)

    def __repr__(self) -> str:
        return "Tanh()"
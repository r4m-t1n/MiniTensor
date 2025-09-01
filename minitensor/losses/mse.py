from minitensor.backend import get_backend
from minitensor import Tensor

class MSE:
    def __init__(self):
        pass

    def __call__(self, y: Tensor, y_hat: Tensor) -> Tensor:
        backend = get_backend(y_hat.dtype)
        result = backend.mse_loss(y._tensor, y_hat._tensor)

        requires_grad = y.requires_grad or y_hat.requires_grad

        return Tensor._new_tensor(result, y_hat.dtype, requires_grad)
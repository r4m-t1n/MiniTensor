from minitensor.backend import get_backend
from minitensor import Tensor

class MAE:
    def __init__(self):
        pass

    def __call__(self, y: Tensor, y_hat: Tensor) -> Tensor:

        backend = get_backend(y_hat.dtype)

        result = backend.mae_loss(y._tensor, y_hat._tensor)

        loss_tensor = Tensor.__new__(Tensor)
        loss_tensor._tensor = result
        loss_tensor.dtype = y_hat.dtype

        return loss_tensor
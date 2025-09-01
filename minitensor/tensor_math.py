from minitensor.backend import get_backend
from minitensor import Tensor

def math_wrapper(func, *args):
    tensor = args[0]

    if not isinstance(tensor, Tensor):
        raise TypeError("ERROR: Only Tensor input is acceptable")

    tensor_args = [arg._tensor if isinstance(arg, Tensor) else arg for arg in args]
    result = func(*tensor_args)

    output_dtype = 'float32' if 'int' in tensor.dtype else tensor.dtype

    return Tensor._new_tensor(result, output_dtype, tensor.requires_grad)

def sqrt(tensor: Tensor) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.sqrt)

def log(tensor: Tensor) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.log)

def exp(tensor: Tensor) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.exp)

def pow(tensor: Tensor, exponent: float) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.pow, tensor, exponent)

def sin(tensor: Tensor) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.sin)

def cos(tensor: Tensor) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.cos)

def tan(tensor: Tensor) -> Tensor:
    backend = get_backend(tensor.dtype)
    return math_wrapper(backend.tan)
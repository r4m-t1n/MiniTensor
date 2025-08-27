from . import minitensor_cpp as mtc

DTYPE_BACKENDS = {
    'float32': mtc.float32,
    'float': mtc.float32,
    'float64': mtc.float64,
    'double': mtc.float64,
    'int32': mtc.int32,
    'int': mtc.int32
}

def is_dtype_valid(dtype: str):
    if dtype not in DTYPE_BACKENDS:
        raise TypeError(f"ERROR: Unsupported data type '{dtype}'.")
    return True

def get_backend(dtype: str):
    if is_dtype_valid(dtype):
        return DTYPE_BACKENDS[dtype]
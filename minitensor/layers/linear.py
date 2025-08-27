from typing import Generator
from minitensor.backend import get_backend
from minitensor import Tensor

class Linear:
    def __init__(self,
        input_features: int,
        output_features: int,
        dtype: str = "float32",
        weight_init = None,
        bias_init = None
        ):

        self.input_f = input_features
        self.output_f = output_features
        self.dtype = dtype
        
        self.backend = get_backend(self.dtype)

        if weight_init is None:
            if 'float' in self.dtype or 'double' in self.dtype:
                weight_init = self.backend.HeNormal()
            else:
                weight_init = self.backend.Constant(1)
        
        if bias_init is None:
            bias_init = self.backend.Constant(0.0 if 'float' in self.dtype or 'double' in self.dtype else 0)

        self._linear = self.backend.Linear(self.input_f, self.output_f, weight_init, bias_init)
        
        self._params = self._linear.parameters()

    def forward(self, x: Tensor) -> Tensor:
        cpp_input = x._tensor

        cpp_result = self._linear.forward(cpp_input)

        new_output_tensor = Tensor.__new__(Tensor)
        new_output_tensor._tensor = cpp_result
        new_output_tensor.dtype = self.dtype

        return new_output_tensor


    @property
    def weight(self) -> Tensor:
        return self._params[0]

    @property
    def bias(self) -> Tensor:
        return self._params[1]

    def parameters(self) -> Generator[Tensor, None, None]:
        yield from self._params
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
        
    def __repr__(self):
        return f"Linear(input_features={self.input_f}, output_features={self.output_f}, dtype='{self.dtype}')"
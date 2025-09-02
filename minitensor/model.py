from typing import Generator
from .tensor import Tensor
from .layers import Linear

class Model:
    def forward(self, *args):
        raise NotImplementedError

    def parameters(self) -> Generator[Tensor, None, None]:
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, (Linear, Model)):
                if hasattr(attr, 'parameters'):
                    yield from attr.parameters()

    def __call__(self, *args) -> Tensor:
        return self.forward(*args)
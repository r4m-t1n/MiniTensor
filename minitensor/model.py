from typing import Generator
from .tensor import Tensor

class Module:
    def forward(self, *args):
        raise NotImplementedError

    def parameters(self) -> Generator[Tensor, None, None]:
        yield from ()

    def __call__(self, *args) -> Tensor:
        return self.forward(*args)

class Sequential(Module):
    def __init__(self, *layers):
        if not all(isinstance(layer, Module) for layer in layers):
            raise TypeError("ERROR: All inputs to Sequential must be instances of Module (layers or activations).")
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
        
    def parameters(self) -> Generator[Tensor, None, None]:
        for layer in self.layers:
            yield from layer.parameters()
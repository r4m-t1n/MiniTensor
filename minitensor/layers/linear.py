import os
import sys
from typing import Generator

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import minitensor_cpp as mtc
from minitensor import Tensor

class Linear:
    def __init__(self,
        input_features: int, output_features: int,
        activation: str = None):
        self.input_f = input_features
        self.output_f = output_features

        self._linear = mtc.Linear(
            self.input_f, self.output_f, activation)
    
    @property
    def weight(self) -> Tensor:
        return self._linear.weight

    @property
    def bias(self) -> Tensor:
        return self._linear.bias

    def parameters(self) -> Generator[Tensor]:
        for param in self._linear.parameters():
            yield param
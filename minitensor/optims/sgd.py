from typing import List
from minitensor import Tensor

class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                updated_tensor = p - (self.lr * p.grad)
                p._tensor = updated_tensor._tensor

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()
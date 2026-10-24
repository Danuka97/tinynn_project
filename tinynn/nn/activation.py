# tinynn/nn/activation.py
import torch
from .base import MinimalModule

class ReLU(MinimalModule):
    def __init__(self):
        self.inputs_memory = None

    def forward(self, X):
        self.inputs_memory = X
        # torch.clamp replaces all numbers below a minimum (0) with that minimum
        return torch.clamp(X, min=0.0)

    def backward(self, dY):
        X = self.inputs_memory
        dX = dY.clone()
        # The Gate: If the original input was 0 or less, stop the gradient
        dX[X <= 0] = 0.0 
        return dX


class LeakyReLU(MinimalModule):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope
        self.inputs_memory = None

    def forward(self, X):
        self.inputs_memory = X
        # torch.where(condition, if_true, if_false)
        return torch.where(X > 0, X, X * self.negative_slope)

    def backward(self, dY):
        X = self.inputs_memory
        dX = dY.clone()
        # The Leaky Gate: If negative, let a tiny bit of gradient through
        dX[X <= 0] *= self.negative_slope
        return dX
# tinynn/nn/sequential.py
from .base import MinimalModule

class Sequential(MinimalModule):
    def __init__(self, *layers):
        # This stores our layers in a list, e.g., [linear1, relu, linear2]
        self.layers = layers

    def parameters(self):
        # Gather all parameters from every layer for the optimizer
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads

    def forward(self, X):
        out = X
        # The Forward Relay Race
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def backward(self, dY):
        dX = dY
        # The Backward Relay Race
        for layer in reversed(self.layers):
            dX = layer.backward(dX) 
        return dX
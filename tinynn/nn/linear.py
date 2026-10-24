# tinynn/nn/linear.py
import torch
from .base import MinimalModule

class LinearLayer(MinimalModule):
    def __init__(self, in_features, out_features, device):
        self.weight = torch.randn(in_features, out_features, device=device) * 0.01
        self.bias = torch.zeros(out_features, device=device)
        
        self.weight_grad = torch.zeros_like(self.weight)
        self.bias_grad = torch.zeros_like(self.bias)
        self.inputs_memory = None 

    def parameters(self):
        # We return the actual tensors here so the optimizer can find them
        return [self.weight, self.bias]

    def grads(self):
        return [self.weight_grad, self.bias_grad]

    def forward(self, X):
        self.inputs_memory = X
        return X @ self.weight + self.bias

    def backward(self, dY):
        X = self.inputs_memory
        # Flatten any leading batch/sequence dims so weight_grad stays (in_features, out_features)
        X_flat = X.reshape(-1, X.shape[-1])
        dY_flat = dY.reshape(-1, dY.shape[-1])
        self.weight_grad = X_flat.transpose(-2, -1) @ dY_flat
        self.bias_grad = dY_flat.sum(dim=0)

        dX = dY @ self.weight.transpose(-2, -1)
        return dX
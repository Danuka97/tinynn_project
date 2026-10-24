# tinynn/nn/embed.py
import torch
from .base import MinimalModule

class EmbeddingLayer(MinimalModule):
    def __init__(self, vocab_size, embed_size, device):
        self.weight = torch.randn(vocab_size, embed_size, device=device) * 0.01
        self.weight_grad = torch.zeros_like(self.weight)
        self.inputs_memory = None

    def parameters(self):
        return [self.weight]

    def grads(self):
        return [self.weight_grad]

    def forward(self, token_ids):
        # Save the token IDs so we know which rows to update later
        self.inputs_memory = token_ids
        
        # PyTorch advanced indexing grabs all the correct rows instantly
        return self.weight[token_ids]

    def backward(self, dY):
        token_ids = self.inputs_memory

        # Flatten to handle any leading batch/sequence dims, and use index_add_
        # so repeated token ids correctly accumulate rather than overwrite.
        flat_ids = token_ids.reshape(-1)
        flat_dY = dY.reshape(-1, dY.shape[-1])
        self.weight_grad.zero_()
        self.weight_grad.index_add_(0, flat_ids, flat_dY)

        # Embeddings are the very first layer, so there is no dX to pass backward!
        return None
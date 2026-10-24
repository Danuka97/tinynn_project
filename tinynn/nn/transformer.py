# tinynn/nn/transformer.py
from .base import MinimalModule
from .attention import SelfAttention
from .linear import LinearLayer
from .activation import ReLU

class TransformerBlock(MinimalModule):
    def __init__(self, embed_size, device):
        self.attention = SelfAttention(embed_size, device)
        
        # Our Feed-Forward Network (FFN)
        self.linear1 = LinearLayer(embed_size, embed_size * 4, device)
        self.relu = ReLU()
        self.linear2 = LinearLayer(embed_size * 4, embed_size, device)

    def ffn(self, x):
        return self.linear2(self.relu(self.linear1(x)))

    def parameters(self):
        return (self.attention.parameters() + self.linear1.parameters() +
                self.linear2.parameters())

    def grads(self):
        return (self.attention.grads() + self.linear1.grads() +
                self.linear2.grads())

    def forward(self, X):
        # 1. Attention + First Residual Connection
        x_attn = X + self.attention(X)

        # 2. Process through the FFN
        ffn_out = self.ffn(x_attn)

        # 3. Second Residual Connection
        out = x_attn + ffn_out

        return out

    def backward(self, dOut):
        # 3. backward through second residual: out = x_attn + ffn_out
        d_x_attn = dOut.clone()
        d_ffn_out = dOut

        # 2. backward through FFN: ffn_out = linear2(relu(linear1(x_attn)))
        d = self.linear2.backward(d_ffn_out)
        d = self.relu.backward(d)
        d = self.linear1.backward(d)
        d_x_attn = d_x_attn + d

        # 1. backward through first residual: x_attn = X + attention(X)
        dX = d_x_attn.clone()
        dX = dX + self.attention.backward(d_x_attn)

        return dX
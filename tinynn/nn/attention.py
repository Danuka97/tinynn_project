# tinynn/nn/attention.py
import torch
from .base import MinimalModule
from .linear import LinearLayer

class SelfAttention(MinimalModule):
    def __init__(self, embed_size, device):
        # We need 3 separate linear layers to project our embeddings
        # Since we want the output vectors to be the same size as the input,
        # in_features and out_features are both 'embed_size'
        self.q_proj = LinearLayer(embed_size, embed_size, device)
        self.k_proj = LinearLayer(embed_size, embed_size, device)
        self.v_proj = LinearLayer(embed_size, embed_size, device)
        
        self.inputs_memory = None

    def parameters(self):
        # Gather all the parameters from our 3 sub-layers for the optimizer
        return self.q_proj.parameters() + self.k_proj.parameters() + self.v_proj.parameters()

    def grads(self):
        return self.q_proj.grads() + self.k_proj.grads() + self.v_proj.grads()

    def forward(self, X):
        self.inputs_memory = X
        
        # 1. Generate Q, K, V
        Q = self.q_proj.forward(X)
        K = self.k_proj.forward(X)
        V = self.v_proj.forward(X)
        
        # 2. Raw scores
        attention_scores = Q @ K.transpose(-2, -1)

        # 3. Scale the scores (preventing them from getting too large)
        d_k = Q.shape[-1]
        scaled_scores = attention_scores / (d_k ** 0.5)

        # 4. masking
        T = X.shape[-2]

        mask = torch.tril(torch.ones(T, T, device=X.device))
        scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
        

        # 4. Softmax (turn into probabilities)
        attention_weights = torch.softmax(scaled_scores, dim=-1)

        # 5. Extract the final context!
        final_output = attention_weights @ V

        # Cache everything backward() needs
        self.Q, self.K, self.V = Q, K, V
        self.attention_weights = attention_weights
        self.d_k = d_k

        return final_output

    def backward(self, dY):
        Q, K, V = self.Q, self.K, self.V
        attention_weights = self.attention_weights
        d_k = self.d_k

        # 5. backward: final_output = attention_weights @ V
        d_attention_weights = dY @ V.transpose(-2, -1)
        dV = attention_weights.transpose(-2, -1) @ dY

        # 4. backward through softmax (dim=-1)
        # dscores = attn_weights * (d_attention_weights - sum(d_attention_weights * attn_weights, dim=-1, keepdim=True))
        weighted_sum = (d_attention_weights * attention_weights).sum(dim=-1, keepdim=True)
        dscaled_scores = attention_weights * (d_attention_weights - weighted_sum)

        # masked (-inf) positions contribute zero softmax weight already, so their grad is ~0

        # 3. backward through scaling
        dattention_scores = dscaled_scores / (d_k ** 0.5)

        # 2. backward through attention_scores = Q @ K.T
        dQ = dattention_scores @ K
        dK = dattention_scores.transpose(-2, -1) @ Q

        # 1. backward through the three projections
        dX_q = self.q_proj.backward(dQ)
        dX_k = self.k_proj.backward(dK)
        dX_v = self.v_proj.backward(dV)

        return dX_q + dX_k + dX_v
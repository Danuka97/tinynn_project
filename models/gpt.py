# models/gpt.py
import torch
from tinynn.nn.base import MinimalModule
from tinynn.nn.embed import EmbeddingLayer
from tinynn.nn.transformer import TransformerBlock
from tinynn.nn.sequential import Sequential
from tinynn.nn.linear import LinearLayer

class GPT(MinimalModule):
    def __init__(self, vocab_size, max_seq_len, embed_size, num_layers, device):
        # The longest sequence the position table has rows for
        self.max_seq_len = max_seq_len

        # 1. The two lookup tables
        self.word_emb = EmbeddingLayer(vocab_size, embed_size, device)
        self.pos_emb = EmbeddingLayer(max_seq_len, embed_size, device)
        
        # 2. Stack of Transformer Blocks
        blocks = [TransformerBlock(embed_size, device) for _ in range(num_layers)]
        self.blocks = Sequential(*blocks)
        
        # 3. Final layer to map back to vocabulary words
        self.lm_head = LinearLayer(embed_size, vocab_size, device)

    def parameters(self):
        return (self.word_emb.parameters() + self.pos_emb.parameters() +
                self.blocks.parameters() + self.lm_head.parameters())

    def grads(self):
        return (self.word_emb.grads() + self.pos_emb.grads() +
                self.blocks.grads() + self.lm_head.grads())

    def forward(self, token_ids):
        # Get the sequence length (token_ids is [batch, seq_len] or just [seq_len])
        T = token_ids.shape[-1]
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len ({self.max_seq_len})")
        
        # Generate position IDs: [0, 1, 2, ..., T-1]
        pos_ids = torch.arange(T, device=token_ids.device)
        
        # Look up the two sets of vectors
        word_vectors = self.word_emb(token_ids)
        pos_vectors = self.pos_emb(pos_ids)
        
        # Combine them!
        X = word_vectors + pos_vectors
        
        # Pass through the transformer blocks
        x = self.blocks(X)
        
        # Predict the next word
        logits = self.lm_head(x)
        return logits

    def backward(self, dLogits):
        # Reverse of forward: lm_head -> blocks -> (word_emb, pos_emb)
        dx = self.lm_head.backward(dLogits)
        dX = self.blocks.backward(dx)

        # X = word_vectors + pos_vectors, so the incoming grad is shared
        self.word_emb.backward(dX)
        # pos_vectors was broadcast across the batch dim (if there is one),
        # so sum its grad back down to [seq_len, embed_size]
        self.pos_emb.backward(dX.reshape(-1, *dX.shape[-2:]).sum(dim=0))
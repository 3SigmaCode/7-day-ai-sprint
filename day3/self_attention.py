import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, \
            "Embedding size must be divisible by heads"

        # Linear projections for Q, K, V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries= nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final projection
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys   = keys.reshape(N, key_len, self.heads, self.head_dim)
        query  = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply the linear projections
        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # Q · Kᵀ → attention scores
        # query shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", query, keys)

        # Scale by sqrt(d_k) for stability
        energy = energy / math.sqrt(self.head_dim)

        # Mask future tokens (decoder-style)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Softmax → probabilities
        attention = torch.softmax(energy, dim=-1)

        # Weighted sum of values
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhqk,nvhd->nqhd", attention, values)

        # Concatenate heads
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out)

if __name__ == "__main__":
    embed_size = 256
    heads = 8
    model = SelfAttention(embed_size, heads)

    x = torch.tensor([[[1.0]*embed_size, [2.0]*embed_size]])

    out = model(x, x, x, mask=None)

    print("Input:", x.shape)
    print("Output:", out.shape)

    assert x.shape == out.shape
    print("✅ Self-Attention works")

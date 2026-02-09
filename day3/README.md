# Day 3: Self-Attention from Scratch

**The brain of the Transformer.**

## What is This?

A pure PyTorch implementation of Multi-Head Self-Attention with **zero abstractions**. No `nn.MultiheadAttention`. Just math.

## The Equation

```
Attention(Q, K, V) = softmax((QK^T) / √d_k) × V
```

## Quick Start

```bash
python self_attention.py
```

Expected output:
```
Input: torch.Size([1, 2, 256])
Output: torch.Size([1, 2, 256])
✅ Self-Attention works
```

## Key Concepts

### 1. **Query, Key, Value (Q, K, V)**
- **Query**: What am I looking for?
- **Key**: What do I offer?
- **Value**: What information do I contain?

### 2. **Multi-Head Attention**
We split the embedding into `heads` independent subspaces:
- Each head attends to different aspects
- Heads run in parallel
- Results are concatenated and projected

### 3. **Scaled Dot-Product**
```python
energy = torch.einsum("nqhd,nkhd->nhqk", query, keys)
energy = energy / math.sqrt(embed_size)  # Prevents gradient explosion
```

### 4. **Causal Masking**
```python
if mask is not None:
    energy = energy.masked_fill(mask == 0, float("-1e20"))
```
Prevents the model from seeing the future during training.

## Architecture

```
[N, seq_len, embed_size]
         ↓
   Split into heads
         ↓
[N, seq_len, heads, head_dim]
         ↓
    Q · K^T / √d_k
         ↓
      Softmax
         ↓
    Attention × V
         ↓
  Concatenate heads
         ↓
[N, seq_len, embed_size]
```

## Parameters

- `embed_size`: Embedding dimension (must be divisible by `heads`)
- `heads`: Number of attention heads (e.g., 8)

## Example Usage

```python
model = SelfAttention(embed_size=256, heads=8)

# Self-attention: Q, K, V are the same
x = torch.randn(1, 10, 256)  # [batch, seq_len, embed_size]
output = model(x, x, x, mask=None)

# Cross-attention: Different Q, K, V
encoder_output = torch.randn(1, 20, 256)
decoder_input = torch.randn(1, 10, 256)
output = model(encoder_output, encoder_output, decoder_input, mask=None)
```

## Why This Matters

**Self-Attention is the core innovation of Transformers.**
- GPT uses it for language generation
- BERT uses it for understanding
- Vision Transformers use it for images

This implementation strips away the magic. It's just linear algebra and you can see every step.

## Next Steps

- **Day 4**: The Speed (Streaming & Latency)

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        # 1. The Attention Layer (The "Social" Layer)
        # We use PyTorch's optimized implementation today.
        # In Day 3, we will rip this open and build it from scratch.
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
        # 2. The Feed Forward Network (The "Thinking" Layer)
        # Expands the dimensions (usually 4x) to process information, then compresses back.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask):
        # --- STEP 1: ATTENTION ---
        # Query, Key, Value shapes: (Seq_Len, Batch_Size, Embed_Size)
        attention_out, _ = self.attention(query, key, value, attn_mask=mask)
        
        # --- STEP 2: ADD & NORM (The Residual Connection) ---
        # CRITICAL: We add the original query to the result.
        # This allows gradients to flow through the network without vanishing.
        x = self.dropout(self.norm1(attention_out + query))
        
        # --- STEP 3: FEED FORWARD ---
        forward_out = self.feed_forward(x)
        
        # --- STEP 4: ADD & NORM AGAIN ---
        out = self.dropout(self.norm2(forward_out + x))
        
        return out

# The Sanity Check
if __name__ == "__main__":
    # Hyperparameters
    EMBED_SIZE = 512
    HEADS = 8
    DROPOUT = 0.1
    FORWARD_EXPANSION = 4
    
    # Dummy Data: (Sequence Length=10, Batch Size=2, Embedding Size=512)
    x = torch.randn(10, 2, EMBED_SIZE)
    
    # Initialize the Block
    block = TransformerBlock(EMBED_SIZE, HEADS, DROPOUT, FORWARD_EXPANSION)
    
    # Forward Pass
    out = block(x, x, x, mask=None)
    
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}") # Must match input
    
    assert x.shape == out.shape, "❌ Shape Mismatch! Check the layers."
    print("✅ Transformer Block Implementation: SUCCESS")

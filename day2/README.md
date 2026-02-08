# Day 2: The Transformer Encoder Block

This directory contains the Python implementation of the **Transformer Encoder Block**, coded from scratch in PyTorch.

This is part of the **3 Sigma AI Engineering Sprint**.

## 🚀 What We Built
We broke down the "Attention Is All You Need" paper (2017) and implemented:
1.  **Multi-Head Attention** (using `nn.MultiheadAttention` for Day 2, manual implementation in Day 3).
2.  **Layer Normalization** (Pre-Norm logic).
3.  **Feed-Forward Network** (Expansion factor of 4).
4.  **Residual Connections** (The Gradient Superhighway).

## 📂 File Structure
- `block.py`: Complete implementation of the `TransformerBlock` class.

## 🛠️ Usage
1.  **Install PyTorch**:
    ```bash
    pip install torch
    ```
2.  **Run the Sanity Check**:
    ```bash
    python3 block.py
    ```
    You should see:
    ```
    Input Shape:  torch.Size([10, 2, 512])
    Output Shape: torch.Size([10, 2, 512])
    ✅ Transformer Block Implementation: SUCCESS
    ```

## 🎥 Watch the Deep Dive
[Link to Video] - In this video, we explain the "why" behind every line of code.

---
*Subscribe to become a 3 Sigma AI Engineer.*

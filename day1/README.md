# Day 1: The Roadmap & Environment Setup 🗺️

Welcome to **Day 1** of the 3 Sigma AI Engineering Sprint.
Today is about **Mindset** and **Environment**. We stop watching tutorials and start building our laboratory.

## 🎯 The Assignment

1.  **Stop Browsing**: Commit to building, not just watching.
2.  **Read the Paper**: Download [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Read the Abstract and Intro.
3.  **Setup your Engine**: Run the `check_acceleration.py` script to verify you are ready for Deep Learning.

## 🛠️ Setup Instructions

### 1. Install Conda
We use Conda to manage our Python environments. If you don't have it, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Create the Environment
Open your terminal and run:

```bash
conda create -n ai_sprint python=3.10
conda activate ai_sprint
```

### 3. Install PyTorch
Visit [pytorch.org](https://pytorch.org/) to get the command for your specific OS.
Common Examples:

**Mac (Silicon):**
```bash
pip install torch torchvision torchaudio
```

**Windows/Linux (NVIDIA CUDA):**
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Acceleration
Run the script included in this folder:

```bash
python day1/check_acceleration.py
```

If you see `🚀 CUDA is AVAILABLE!` or `🚀 MPS is AVAILABLE!`, you are ready for **Day 2: Transformer Internals**.

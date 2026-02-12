# ⚡️ 7-Day AI Engineering Sprint

Welcome to the **7-Day AI Engineering Sprint**. This repository contains the source code, scripts, and educational materials for building production-grade AI systems from scratch.

Each day focuses on a critical component of the modern AI stack, moving from theory to production engineering.

## 📅 Curriculum

### Day 1: The Foundation
**Theme:** Deep Dive into AI Engineering
- 📂 **Source:** `src/render_day1_deep_dive.py`
- 🎯 **Goal:** Setting up the environment, understanding the landscape, and preparing for the sprint.

### Day 2: Transformers from Scratch
**Theme:** The Architecture that Changed Everything
- 📂 **Source:** `day2_transformer/`
- 🎯 **Goal:** coding a Transformer model from scratch (no PyTorch/TensorFlow) to understand the math.

### Day 3: Self-Attention Mechanism
**Theme:** The "Brain" of the Transformer
- 📂 **Source:** `day3_attention/`
- 🎯 **Goal:** Visualizing and implementing the Query, Key, Value attention mechanism.

### Day 4: Streaming & Latency
**Theme:** Real-Time AI Systems
- 📂 **Source:** `day4_streaming/`
- 🎯 **Goal:** Building low-latency streaming pipelines with FastAPI and WebSockets.

### Day 5: Production RAG
**Theme:** Retrieval Augmented Generation
- 📂 **Source:** `day5_prod_rag/`
- 🎯 **Goal:** Building a robust RAG pipeline with hybrid retrieval (Vector + BM25) and grounding.

### Day 6: RAG Evaluation (The Judge)
**Theme:** "But would you ship it?"
- 📂 **Source:** `output/rag_eval_github/` (Release Package)
- 🎯 **Goal:** Building an automated "Judge" system to evaluate RAG accuracy using a Golden Dataset.

### Day 7: The Guardrails (Upcoming)
**Theme:** Security & Safety
- 🎯 **Goal:** Prompt injection defense, firewalls, and making your AI hack-proof.

## 🛠️ Usage

This repository is structured as a Content Engine.
- **`src/`**: Contains video rendering scripts (Manim/MoviePy) for generating the educational content.
- **`dayX_*/`**: Contains the standalone code examples for each day.

To run the RAG Judge from Day 6:
```bash
cd output/rag_eval_github
pip install -r requirements.txt # (if available) or pip install openai
python rag_judge.py
```

## 📺 Follow the Sprint
Subscribe to follow the daily releases and deep dives.

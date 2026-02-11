# Day 5: Production-Grade RAG (From Scratch)

This is the code companion for the **Day 5** video. It implements a complete, production-grade Retrieval Augmented Generation (RAG) pipeline **without** using heavy frameworks like LangChain or LlamaIndex. 

We build it step-by-step to ensure full control and transparency.

## 🚀 The Stack

- **pypdf**: For robust, low-level PDF text extraction.
- **sentence-transformers**: To generate dense vector embeddings (the "meaning").
- **chromadb**: A lightweight vector database to store our semantic memory.
- **rank-bm25**: For keyword retrieval (finding exact matches like IDs or acronyms).
- **openai**: For generation (GPT-4o-mini).

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Data**:
   Place a PDF file named `data.pdf` in this directory.

3. **Set API Key**:
   Export your OpenAI API Key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. **Run**:
   ```bash
   python main.py
   ```

## 🧠 Core Concepts (Why we do this)

### 1. Recursive Chunking with Overlap
**Why?** If you just split text every 800 characters, you might cut a sentence in half.
**The Fix:** We use an `overlap` (e.g., 100 chars). This ensures that every thought is preserved in at least one chunk.

### 2. Hybrid Retrieval (The "Secret Sauce")
**Why?** Vector search is great for *meaning* ("dog" matches "puppy"), but bad for *exact words* (product codes, specific names).
**The Fix:** We combine **Semantic Search** (Vectors) with **Keyword Search** (BM25).
- Vectors find concepts.
- BM25 finds exact matches.
- Together, they give you high recall and high precision.

### 3. Systematic Prompt Injection
**Why?** AI models hallucinate. They make things up.
**The Fix:** We strictly instruct the model: *"Answer ONLY from the context below."* This grounds the AI in reality.

## 🔮 What's Next? (Day 6)
In Day 6, we will add **Reranking** (Cross-Encoders) to sort our retrieved results by relevance, drastically improving accuracy.

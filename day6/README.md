# RAG Evaluator (The Judge)

This repository contains the source code for the "RAG Judge" system demonstrated in the video: **"Your RAG System Works... But Would You Ship It?"**

## 🚀 Overview

Most RAG systems fail in production because they lack rigorous evaluation. This project implements an **LLM-as-a-Judge** pattern to automatically evaluate the quality of your RAG pipeline's answers against a "Golden Dataset".

### Features
- **Dataset Loading**: Loads question/ground_truth pairs from `dataset.json`.
- **RAG Simulation**: A mock `run_rag` function (replace with your actual pipeline).
- **The Judge**: Uses GPT-4 to score answers on Accuracy, Relevance, and Completeness.
- **Pass/Fail Logic**: Automatically flags answers that drop below a defined threshold (e.g., 6/10).
- **Alerting**: Simulates a production alert for failing grades.

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install openai
   ```

2. **Set API Key**:
   Export your OpenAI API key or set it in the script.
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. **Run the Judge**:
   ```bash
   python rag_judge.py
   ```

## 📂 Structure

- `rag_judge.py`: Main script containing the RAG logic, Judge prompt, and correct loop.
- `dataset.json`: A sample "Golden Dataset" with ground truth answers.

## ⚠️ Note

The `run_rag` function in this repo is a simulation. In a real deployment, you would replace this function with your actual retrieval (Pinecone, Weaviate, etc.) and generation logic.

## 📺 Connect

If you found this useful, check out the full breakdown on YouTube.
**Follow & Subscribe** for more production AI engineering content.

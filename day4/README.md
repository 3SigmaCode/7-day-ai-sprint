# Day 4 – Streaming & Latency

This module converts a blocking LLM response into a streaming response
using FastAPI and Python generators.

## Why Streaming?
Users don’t care when an answer finishes.
They care when it starts.

Streaming improves perceived latency without changing the model.

## How It Works
- The LLM emits tokens incrementally
- A Python generator yields tokens
- FastAPI streams them to the client

## Run
uvicorn app:app --reload

## Next
Day 5: Adding memory using Retrieval-Augmented Generation (RAG)

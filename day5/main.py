import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ==========================================
# 1. Load PDF (Raw Text Extraction)
# ==========================================
def load_pdf(path):
    print(f"📄 Loading PDF: {path}...")
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ==========================================
# 2. Recursive Chunking (with Overlap)
# ==========================================
def chunk_text(text, chunk_size=800, overlap=100):
    print("✂️ Chunking text...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward, but backtrack by overlap to preserve context
        start += chunk_size - overlap
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks

# ==========================================
# 3. Embeddings & Storage
# ==========================================
def setup_vector_db(chunks):
    print("🧠 Generating Embeddings & Storing in Chroma...")
    
    # Initialize Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks)
    
    # Initialize Vector DB
    client = chromadb.Client()
    # Delete if exists to start fresh
    try:
        client.delete_collection("rag_prod")
    except:
        pass
        
    collection = client.create_collection("rag_prod")
    
    # Add to Chroma
    collection.add(
        documents=chunks,
        embeddings=chunk_embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )
    
    return collection, model

# ==========================================
# 4. Keyword Index (BM25)
# ==========================================
def setup_bm25(chunks):
    print("📚 Building BM25 Index...")
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

# ==========================================
# 5. Hybrid Retrieval
# ==========================================
def hybrid_search(query, collection, model, bm25, chunks, top_k=5):
    print(f"🔍 Searching for: '{query}'")
    
    # 1. Semantic Search (Vector)
    query_embedding = model.encode([query])
    vector_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    semantic_docs = vector_results["documents"][0]
    
    # 2. Keyword Search (BM25)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top_k indices
    top_bm25_indices = np.argsort(bm25_scores)[-top_k:]
    keyword_docs = [chunks[i] for i in top_bm25_indices]
    
    # 3. Combine & Deduplicate
    combined_docs = list(set(semantic_docs + keyword_docs))
    
    return combined_docs[:top_k]

# ==========================================
# 6. Generation (Prompt Injection)
# ==========================================
def generate_answer(query, context_docs):
    print("🤖 Generating Answer...")
    
    context = "\n\n".join(context_docs)
    
    prompt = f"""
You are a factual assistant.
Answer ONLY from the context below.

Context:
{context}

Question:
{query}

Answer:
"""
    
    # Initialize OpenAI (Assumes OPENAI_API_KEY is in env)
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    # Check for PDF
    pdf_path = "data.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  File '{pdf_path}' not found. Please add a PDF to this directory.")
        # Create a dummy PDF for demonstration if needed, or just exit
        exit()

    # 1. Load
    raw_text = load_pdf(pdf_path)
    
    # 2. Chunk
    chunks = chunk_text(raw_text)
    
    # 3. Store (Vector)
    collection, model = setup_vector_db(chunks)
    
    # 4. Index (Keyword)
    bm25 = setup_bm25(chunks)
    
    # 5. Retrieve
    query = "What is the revenue growth strategy?" # Change this to your query
    context_docs = hybrid_search(query, collection, model, bm25, chunks)
    
    # 6. Generate
    answer = generate_answer(query, context_docs)
    
    print("\n💡 Answer:")
    print(answer)

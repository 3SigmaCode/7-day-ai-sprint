import os
import json
import openai
from typing import List, Dict, Any

# --- Configuration ---
# 1. Set your OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "sk-..." 
client = openai.OpenAI()

PASS_THRESHOLD = 6.0
DATASET_PATH = "dataset.json"

# --- 1. The RAG System (System Under Test) ---
def retrieve_context(question: str) -> str:
    """
    Simulates a retrieval system. 
    In production, this would query a Vector DB (Pinecone, Weaviate) + BM25 (Elasticsearch).
    """
    # Mock context for demonstration
    knowledge_base = {
        "hybrid": "Hybrid retrieval combines vectors and keywords.",
        "RAG": "RAG retrieves docs to ground LLMs.",
        "Judge": "A Judge is an LLM that evaluates other LLMs."
    }
    
    # Simple keyword match simulation
    retrieved = []
    for key, value in knowledge_base.items():
        if key.lower() in question.lower():
            retrieved.append(value)
            
    if not retrieved:
        return "No relevant context found."
    
    return "\n".join(retrieved)

def run_rag(question: str) -> str:
    """
    The actual RAG pipeline: Retrieve -> Augment -> Generate
    """
    context = retrieve_context(question)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Optimizing for speed/cost in production
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Use this context to answer:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# --- 2. The Judge (Evaluation System) ---
def evaluate_answer(question: str, ground_truth: str, rag_answer: str) -> Dict[str, int]:
    """
    Uses a stronger model to evaluate the RAG answer against the Ground Truth.
    Returns a dictionary of scores.
    """
    prompt = f"""
    You are an expert AI Evaluator.

    Question:
    {question}

    Ground Truth:
    {ground_truth}

    RAG Answer:
    {rag_answer}

    Compare the RAG Answer to the Ground Truth.
    Score 1-10 for:
    1. Accuracy (Is it factually correct?)
    2. Relevance (Is it concise and related to the user query?)
    3. Completeness (Does it cover all parts of the ground truth?)

    Return ONLY a JSON object in this format:
    {{
        "accuracy": <int>,
        "relevance": <int>,
        "completeness": <int>
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o", # Stronger model for judging
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(response.choices[0].message.content)
        # Calculate average
        data["average"] = round((data["accuracy"] + data["relevance"] + data["completeness"]) / 3, 1)
        return data
    except Exception as e:
        print(f"Error parsing judge output: {e}")
        return {"accuracy": 0, "relevance": 0, "completeness": 0, "average": 0}

# --- 3. Alert System ---
def trigger_alert(question: str, scores: Dict[str, int], rag_answer: str):
    """
    Simulates a PagerDuty/Slack alert.
    """
    print("\n🚨 [ALERT] RAG Evaluation FAILED")
    print(f"Question: {question}")
    print(f"Answer: {rag_answer}")
    print(f"Scores: {scores}")
    print("-" * 30)

# --- Main Evaluation Loop ---
def run_evaluation_suite():
    print("🚀 Starting RAG Evaluation Suite...\n")
    
    # Load Golden Dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    results = []
    
    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"Testing: {question}...")
        
        # 1. Run System
        rag_answer = run_rag(question)
        
        # 2. Evaluate
        scores = evaluate_answer(question, ground_truth, rag_answer)
        
        # 3. Log Result
        result = {
            "question": question,
            "answer": rag_answer,
            "scores": scores,
            "pass": scores["average"] >= PASS_THRESHOLD
        }
        results.append(result)
        
        # 4. Check Pass/Fail
        if not result["pass"]:
            trigger_alert(question, scores, rag_answer)
        else:
            print(f"✅ PASS (Score: {scores['average']}/10)")
            
    # Final Report
    passed_count = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\n📊 Summary: {passed_count}/{total} Passed ({(passed_count/total)*100:.1f}%)")

if __name__ == "__main__":
    run_evaluation_suite()

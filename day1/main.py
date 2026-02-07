import os
from litellm import completion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 3 SIGMA CONFIG: CAPABILITY REGISTRY
models = {
    "fast": "groq/llama3-8b-8192",      # Latency: <200ms
    "smart": "gpt-4o",                 # Intelligence: High
    "cheap": "deepseek/deepseek-chat"  # Cost: Low
}

def universal_chat(prompt, capability="smart"):
    """
    The Universal Router.
    Switches between providers based on the 'capability' flag.
    """
    print(f"🔄 Routing to: {models[capability]}...")
    
    try:
        response = completion(
            model=models[capability],
            messages=[{"role": "user", "content": prompt}],
            # AUTOMATIC FALLBACKS (The Safety Net)
            fallbacks=["gpt-3.5-turbo", "claude-3-haiku"],
            timeout=10
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ System Failure: {str(e)}"

if __name__ == "__main__":
    # Test 1: Cheap (DeepSeek)
    print("\n--- TEST: CHEAP ---")
    print(universal_chat("Explain the Adapter Pattern in 1 sentence.", capability="cheap"))

    # Test 2: Smart (GPT-4o)
    print("\n--- TEST: SMART ---")
    print(universal_chat("Write a Haiku about API Latency.", capability="smart"))
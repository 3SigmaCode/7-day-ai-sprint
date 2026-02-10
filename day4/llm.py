def stream_tokens(prompt: str):
    try:
        for token in llm.stream(prompt):
            yield token
    except Exception as e:
        yield f"\n[ERROR]: {str(e)}"

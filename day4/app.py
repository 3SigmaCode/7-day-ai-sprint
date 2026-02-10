from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm import stream_tokens

app = FastAPI()

@app.post("/chat")
def chat(prompt: str):
    return StreamingResponse(
        stream_tokens(prompt),
        media_type="text/plain"
    )

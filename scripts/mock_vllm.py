"""Mock OpenAI-compatible vLLM server for local testing (no GPU required).

Implements the same endpoints the API gateway calls:
  POST /v1/chat/completions
  GET  /health
  GET  /v1/models
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
import hashlib

app = FastAPI(title="Mock vLLM (local)")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 256


def mock_answer(prompt: str) -> str:
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return (
        f"[mock-vllm answer {h}] Based on the provided context, here is a concise response: "
        f"platform engineering is the discipline of building internal developer platforms "
        f"that improve developer experience and productivity through self-service capabilities, "
        f"golden paths, and automation. (mocked locally for lab28)"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    answer = mock_answer(last_user)
    return {
        "id": f"chatcmpl-mock-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": len(last_user)//4, "completion_tokens": len(answer)//4, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
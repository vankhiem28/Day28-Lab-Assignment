# api-gateway/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import httpx, os, time

app = FastAPI(title="AI Platform API Gateway")
Instrumentator().instrument(app).expose(app)  # Integration 9: Prometheus

VLLM_URL = os.environ["VLLM_URL"]
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    embedding: list[float] = Field(default_factory=lambda: [0.0] * 384)


@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    start = time.time()
    try:
        # 1. Vector search
        async with httpx.AsyncClient(timeout=10) as client:
            search_resp = await client.post(
                f"{QDRANT_URL}/collections/documents/points/search",
                json={"vector": req.embedding, "limit": 3},
            )
            context = search_resp.json().get("result", [])

        # 2. LLM inference
        prompt = f"Context: {context}\n\nQuery: {req.query}"
        async with httpx.AsyncClient(timeout=30) as client:
            llm_resp = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            llm_resp.raise_for_status()
            result = llm_resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e

    latency = (time.time() - start) * 1000
    return {
        "answer": result["choices"][0]["message"]["content"],
        "latency_ms": round(latency, 2),
        "model": result.get("model", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}

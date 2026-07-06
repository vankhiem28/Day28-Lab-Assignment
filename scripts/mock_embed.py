"""Mock embedding service for local testing.

Implements: POST /embed -> {"embeddings": [[...], ...]} with 384-dim vectors.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import hashlib

app = FastAPI(title="Mock Embedding Service (local)")


class EmbedRequest(BaseModel):
    texts: List[str]


def text_to_vector(text: str, dim: int = 384) -> List[float]:
    """Deterministic 384-dim vector from text hash, normalised to unit length."""
    h = hashlib.sha256(text.encode()).digest()
    raw = [(b - 128) / 128.0 for b in (h * 16)[:dim]]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    return {"embeddings": [text_to_vector(t) for t in req.texts]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
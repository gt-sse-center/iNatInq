"""Lightweight SigLIP2 embedding server on Apple MPS.

Drop-in replacement for Infinity's /embeddings API so InfinityClient works unchanged.
Loads google/siglip2-so400m-patch14-384 on MPS (Metal) GPU for fast inference.

Usage:
    uv run python bench/tools/siglip2_server.py [--port 7997] [--model google/siglip2-so400m-patch14-384]
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("siglip2-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
_model: Any = None
_processor: Any = None
_model_id: str = ""
_device: torch.device = torch.device("cpu")

# Regex for base64 data URIs: data:<mime>;base64,<data>
_DATA_URI_RE = re.compile(r"^data:[^;]+;base64,(.+)$", re.DOTALL)

VECTOR_SIZES: dict[str, int] = {
    "google/siglip2-so400m-patch14-384": 1152,
    "google/siglip2-base-patch16-224": 768,
    "google/siglip-so400m-patch14-384": 1152,
    "google/siglip-base-patch16-224": 768,
}


# ---------------------------------------------------------------------------
# Request / Response models (match Infinity's OpenAI-compatible format)
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    model: str = "default"
    input: list[str] | str
    modality: str = "text"


class EmbeddingEntry(BaseModel):
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingEntry]
    model: str
    usage: dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"
    capabilities: list[str]


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def _decode_image(data_uri: str) -> Image.Image:
    """Decode a base64 data URI to a PIL Image."""
    match = _DATA_URI_RE.match(data_uri)
    if not match:
        raise HTTPException(status_code=400, detail="Image input must be a base64 data URI")
    raw = base64.b64decode(match.group(1))
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# Embedding logic
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _embed_texts(texts: list[str]) -> list[list[float]]:
    inputs = _processor(
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    out = _model.get_text_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().tolist()


@torch.inference_mode()
def _embed_images(images: list[Image.Image]) -> list[list[float]]:
    inputs = _processor(images=images, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    out = _model.get_image_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().tolist()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _model_id, _device

    _model_id = app.state.model_id
    logger.info("Loading model %s ...", _model_id)

    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", _device)

    _processor = AutoProcessor.from_pretrained(_model_id)
    _model = AutoModel.from_pretrained(_model_id)
    _model.eval()
    _model = _model.to(_device)

    dims = VECTOR_SIZES.get(_model_id, "?")
    logger.info("Model loaded: %s (%s-dim) on %s", _model_id, dims, _device)

    # Warmup pass
    logger.info("Warming up...")
    _ = _embed_texts(["warmup"])
    if _device.type == "mps":
        torch.mps.synchronize()
    logger.info("Ready.")

    yield


app = FastAPI(title="SigLIP2 Embedding Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def models():
    return {
        "data": [
            {
                "id": _model_id,
                "object": "model",
                "owned_by": "local",
                "created": int(time.time()),
                "backend": "torch",
                "capabilities": ["embed", "image_embed"],
                "stats": {"queue_fraction": 0.0, "queue_absolute": 0, "results_pending": 0, "batch_size": 32},
            }
        ],
        "object": "list",
    }


@app.post("/embeddings")
async def embeddings(req: EmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    n = len(inputs)

    if req.modality == "image":
        images = [_decode_image(uri) for uri in inputs]
        vectors = _embed_images(images)
    else:
        vectors = _embed_texts(inputs)

    return EmbeddingResponse(
        data=[EmbeddingEntry(embedding=v, index=i) for i, v in enumerate(vectors)],
        model=_model_id,
        usage={"prompt_tokens": n, "total_tokens": n},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SigLIP2 embedding server (MPS)")
    parser.add_argument("--model-id", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--port", type=int, default=7997)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app.state.model_id = args.model_id
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

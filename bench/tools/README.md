# Benchmark Tools

Scripts for benchmark dataset preparation and embedding. All scripts are standalone and should be run from the repository root.

## Dataset Conversion

### `convert_inquire.py`

Converts raw INQUIRE benchmark data into the project's JSON dataset format.

## Embedding Server

### `siglip2_server.py`

Lightweight FastAPI embedding server for SigLIP2 SO400M on Apple MPS GPU. Drop-in replacement for Infinity's `/embeddings` API — `InfinityClient` works unchanged.

Required because the stock Infinity server (`infinity-emb` 0.0.77) is incompatible with `transformers >= 5.x`.

```bash
uv run python bench/tools/siglip2_server.py \
    --model-id google/siglip2-so400m-patch14-384 \
    --port 7997
```

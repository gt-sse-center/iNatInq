# Plan: INQUIRE Subset Dataset + Benchmark CLI Wiring

## Context

We have a remote Qdrant instance in Azure with 1M indexed iNat images and a hosted CLIP endpoint. The INQUIRE benchmark dataset references iNat photo IDs, but only a subset of those photos exist in our 1M collection. We need to:

1. Generate a filtered INQUIRE dataset containing only queries with 100% relevant-doc coverage in the collection
2. Wire the benchmark CLI `run` and `compare` commands to actually execute searches against the remote providers

**Critical finding from Qdrant inspection:**
- Point IDs are UUID5 strings: `uuid5(NAMESPACE_URL, photo_id)` (e.g., `uuid5(URL, "1316621")` → `000019de-...`)
- Payload `s3_key` field contains the plain numeric photo ID (e.g., `"1316621"`)
- CLIP endpoint uses Azure ML `input_data` format with `columns: ["image", "text"]`
- Existing `CLIPClient` already supports `hosted_clip` backend with `embed_text` / `embed_text_async`

---

## Implementation Steps

### Step 1: ID Mapping Module

**New file:** `src/core/benchmark/id_mapping.py`

`S3KeyIDMapper` (attrs frozen):
- `doc_id_to_point_id(doc_id: str) -> str` — `str(uuid5(NAMESPACE_URL, doc_id))`
- `point_id_to_doc_id(point_id: str, payload: dict) -> str` — reads `payload["s3_key"]`

### Step 2: Search Pipeline

**New file:** `src/core/benchmark/search_pipeline.py`

`CLIPSearchPipeline` (attrs frozen):
- Composes CLIPClient + VectorDBProvider + S3KeyIDMapper
- `async search(query_text, *, collection, limit) -> SearchPipelineResult`

### Step 3: Update DefaultBenchmarkRunner

**Modify:** `src/core/benchmark/runner/default.py`

- Add `search_pipeline` and `collection` constructor params
- Use pipeline for embed→search→map if available, else existing placeholder

### Step 4: Metrics Builder

**New file:** `src/core/benchmark/metrics/builder.py`

- `build_metrics_from_config(config) -> list[Metric]`

### Step 5: Provider Factory

**New file:** `src/core/benchmark/provider_factory.py`

- `resolve_search_pipeline(provider_name, *, collection) -> tuple[VectorDBProvider, CLIPSearchPipeline]`

### Step 6: Wire CLI Commands

**Modify:** `src/core/benchmark/cli.py`

- Wire `run` and `compare` with real providers, metrics builder, reporters

### Step 7: Subset Generator Script

**New file:** `scripts/generate_inquire_subset.py`

- Filter INQUIRE dataset to only queries whose relevant docs exist in Qdrant

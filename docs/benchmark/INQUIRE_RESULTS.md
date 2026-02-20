# INQUIRE Benchmark Results

Evaluation of our semantic image search pipeline against the [INQUIRE benchmark](https://github.com/inquire-benchmark/INQUIRE) (NeurIPS 2024), a text-to-image retrieval benchmark built on iNaturalist.

## Pipeline Under Test

| Component | Configuration |
|-----------|---------------|
| Embedding model | OpenAI CLIP ViT-B/32 (hosted, Azure ML) |
| Vector database | Qdrant Cloud (Azure East US) |
| Collection size | 1,000,000 iNaturalist images |
| ID mapping | UUID5(NAMESPACE_URL, photo_id) |
| Evaluation k | 50 (INQUIRE default) |

## Dataset Preparation

The INQUIRE benchmark provides human-annotated text-to-image retrieval queries over the full iNaturalist photo corpus (~3.6M images). Our Qdrant collection indexes 1M of those images, so not all relevant documents for every query exist in our collection.

We generated subset datasets by:

1. Collecting all unique relevant document IDs across all INQUIRE queries
2. Converting each to its Qdrant point ID via `UUID5(NAMESPACE_URL, photo_id)`
3. Batch-checking existence against the live Qdrant collection
4. **Trimming** each query's relevant list to only documents present in the collection
5. Dropping queries with zero remaining relevant documents

This preserves query diversity while ensuring every relevance judgment can be evaluated. Queries retain their original text and IDs; only the relevant document lists are narrowed.

### Dataset Statistics

| | Full INQUIRE | Our Subset | Coverage |
|---|---|---|---|
| **Val split** | 50 queries / 6,226 relevant | 47 queries / 1,443 relevant | 94% queries, 23% docs |
| **Test split** | 200 queries / 26,470 relevant | 186 queries / 5,782 relevant | 93% queries, 22% docs |

Average relevant documents per query: ~31 (subset) vs ~125 (full).

The subset files are at:
- `benchmarks/inquire/inquire-val-subset.json`
- `benchmarks/inquire/inquire-test-subset.json`

Regenerate with:
```bash
uv run python scripts/generate_inquire_subset.py \
  --input benchmarks/inquire/inquire-val.json \
  --output benchmarks/inquire/inquire-val-subset.json \
  --qdrant-url $QDRANT_URL \
  --qdrant-api-key $QDRANT_API_KEY \
  --collection demo-data-1M_images
```

## Results (Val Split, k=50)

### Retrieval Quality

| Metric | Score | Description |
|--------|------:|-------------|
| **Precision@50** | 0.0834 | 8.3% of the top-50 results are relevant |
| **Recall@50** | 0.1567 | 15.7% of relevant documents appear in the top-50 |
| **MAP** | 0.0679 | Average precision across relevant hits (INQUIRE normalization) |
| **NDCG@50** | 0.1518 | Ranking quality accounting for position of relevant results |
| **MRR** | 0.3224 | On average, the first relevant result appears around rank 3 |

### Latency

| Metric | Value |
|--------|------:|
| **p50** | 154.5 ms |
| **p95** | 240.8 ms |
| **p99** | 316.7 ms |
| **Mean** | 173.3 ms |
| **QPS** | 5.77 |

Latency measured end-to-end: CLIP text embedding (Azure ML) + Qdrant vector search (Azure Cloud). Sequential queries (concurrency=1) from a local client.

## Interpreting the Results

### How these compare to published numbers

The INQUIRE paper (Vendrow et al., NeurIPS 2024, Table 1) reports ViT-B/32 full-rank retrieval over the complete iNaturalist corpus (~3.6M images):

| Metric | INQUIRE Paper (ViT-B/32) | Our Result |
|--------|-------------------------:|----------:|
| MAP | ~0.07–0.10 | 0.068 |
| NDCG@50 | ~0.15–0.20 | 0.152 |

Our results fall within the expected range for ViT-B/32, with two important caveats:

1. **Smaller corpus (1M vs 3.6M)**: Fewer distractors should make retrieval slightly easier, but we also have fewer relevant documents per query (23% of the full relevant set).

2. **Trimmed relevant lists**: Each query has ~31 relevant documents on average instead of ~125. This affects recall (fewer targets to find) and MAP normalization (the denominator `min(|relevant|, k)` changes).

These factors roughly cancel out, which is why our numbers land close to the published range.

### What the metrics tell us

- **MRR of 0.32** means the system typically surfaces a relevant result by rank 3. For an exploratory search interface, this is reasonable — users see something relevant near the top.

- **Precision@50 of 0.08** means ~4 out of 50 results are relevant. In a collection of 1M images with ~31 relevant targets, random chance would yield 0.0031% precision. We are ~27x above random baseline.

- **Recall@50 of 0.16** means we find about 1 in 6 relevant documents in the top-50. Given that some queries have hundreds of relevant images in the full dataset (max 298 in our subset), retrieving all of them in 50 slots is not expected.

- **MAP of 0.07** reflects that relevant results are spread across the top-50 rather than concentrated at the top. This is characteristic of CLIP ViT-B/32, which captures broad semantic similarity but lacks fine-grained discrimination.

- **NDCG@50 of 0.15** accounts for ranking position — relevant results appearing earlier score higher. The gap between NDCG and recall suggests relevant results are reasonably well-ranked when found.

### Metric alignment with INQUIRE

Our evaluation uses the INQUIRE-specific AP normalization:

```
AP = Σ P@rank_hit / min(|relevant|, k)
```

This differs from the standard MAP formula (which divides by `|relevant|`) by capping the denominator at `k`. This avoids penalizing queries where the number of relevant documents exceeds `k` — a query with 200 relevant docs shouldn't be scored differently than one with 50 when evaluating at k=50.

## Running the Benchmark

```bash
./scripts/run_benchmark.sh
```

Or with explicit configuration:

```bash
QDRANT_URL=... QDRANT_API_KEY=... \
CLIP_BACKEND=hosted_clip CLIP_URL=... CLIP_API_KEY=... CLIP_MODEL=ViT-B/32 \
uv run python -m core.benchmark.cli run \
  --dataset benchmarks/inquire/inquire-val-subset.json \
  --provider qdrant \
  --collection demo-data-1M_images \
  --limit 50 --warmup 2
```

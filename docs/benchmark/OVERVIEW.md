# Vector DB Benchmarking Framework

## What Is It?

A provider-agnostic benchmarking framework for evaluating vector database performance using standard Information Retrieval (IR) metrics and latency measurements. The framework enables objective comparison between vector databases (Qdrant, Weaviate, and future providers) using gold standard datasets with known relevance judgments.

## Why It Matters

Vector database selection significantly impacts search quality and system performance. This framework provides:

- **Objective comparison** between providers using industry-standard metrics
- **Reproducible benchmarks** with versioned gold standard datasets
- **Production-ready measurements** including latency percentiles (p50/p95/p99) and throughput (QPS)
- **Graded relevance support** for nuanced quality assessment via NDCG

## Metrics

**Quality Metrics (IR):**

| Metric | What It Measures |
|--------|------------------|
| Precision@K | How many retrieved results are relevant |
| Recall@K | How many relevant items were retrieved |
| MAP | Average precision across all relevant hits |
| MRR | How quickly the first relevant result appears |
| NDCG | Ranking quality with graded relevance scores |

**Performance Metrics:**

| Metric | What It Measures |
|--------|------------------|
| p50/p95/p99 | Latency percentiles in milliseconds |
| QPS | Queries per second throughput |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Gold Standard  │────▶│ BenchmarkRunner │────▶│    Reporters    │
│    Dataset      │     │                 │     │ (Console, JSON) │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │ Qdrant  │  │Weaviate │  │ Future  │
              └─────────┘  └─────────┘  └─────────┘
```

**Key Design Decisions:**

- **Provider-agnostic**: Uses existing `VectorDBProvider` interface—any registered provider works
- **Extensible metrics**: Auto-registration via metaclass; add metrics without modifying core
- **1 to N databases**: Same runner works for single provider or multi-provider comparison

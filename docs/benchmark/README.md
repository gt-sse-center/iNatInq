# Vector DB Benchmarking Framework

A production-grade, cloud-deployable benchmarking framework for evaluating vector database providers on precision/recall metrics and latency characteristics using gold standard datasets.

## Design Principles

1. **DRY** - Shared abstractions via ABCs, mixins, and decorators
2. **Provider-agnostic** - Plug any vector DB via existing `VectorDBProvider` interface
3. **Cloud-native** - Async, stateless, configurable via environment
4. **Observable** - Structured JSON reports for analysis
5. **Extensible** - Add new metrics/reporters without modifying core

## Package Structure

```
src/benchmark/
├── __init__.py
├── metrics/
│   ├── __init__.py
│   ├── base.py              # Metric ABC + MetricRegistry metaclass
│   ├── ir.py                # IR metrics: Precision@K, Recall@K, MAP, NDCG, MRR
│   └── latency.py           # Latency metrics: p50, p95, p99, QPS
├── datasets/
│   ├── __init__.py
│   ├── base.py              # Dataset ABC
│   ├── json_dataset.py      # JSON file loader
│   └── schemas.py           # Pydantic schemas for gold standard format
├── runner/
│   ├── __init__.py
│   ├── base.py              # BenchmarkRunner ABC
│   └── comparison.py        # ComparisonRunner for multi-provider benchmarks
├── reporters/
│   ├── __init__.py
│   ├── base.py              # Reporter ABC
│   ├── console.py           # Pretty console output
│   └── json_reporter.py     # JSON file output
├── decorators.py            # @timed, @benchmark_provider decorators
├── config.py                # Pydantic settings for configuration
└── cli.py                   # Click CLI entry point
```

## Gold Standard Dataset Schema

```json
{
  "name": "inat-image-benchmark-v1",
  "description": "iNaturalist image search relevance judgments",
  "modality": "image",
  "queries": [
    {
      "id": "q001",
      "text": "red bird on branch",
      "relevant": ["img_001", "img_042", "img_103"],
      "graded_relevance": {
        "img_001": 3,
        "img_042": 2,
        "img_103": 1
      }
    }
  ],
  "metadata": {
    "created_at": "2026-01-30",
    "annotators": 3,
    "agreement_threshold": 0.8
  }
}
```

## Metrics

### Information Retrieval Metrics

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K retrieved documents that are relevant |
| **Recall@K** | Fraction of relevant documents that appear in top-K |
| **MAP** | Mean Average Precision - average precision at each relevant hit |
| **NDCG** | Normalized Discounted Cumulative Gain - uses graded relevance |
| **MRR** | Mean Reciprocal Rank - reciprocal of first relevant result's rank |

### Latency Metrics

| Metric | Description |
|--------|-------------|
| **p50** | Median latency (50th percentile) |
| **p95** | 95th percentile latency |
| **p99** | 99th percentile latency |
| **QPS** | Queries per second throughput |

## CLI Usage

```bash
# Run benchmark comparison across providers
uv run python -m src.benchmark.cli compare \
    --dataset benchmarks/inat-gold.json \
    --providers qdrant,weaviate \
    --metrics precision@10,recall@10,map,ndcg \
    --output results.json

# Run single provider benchmark
uv run python -m src.benchmark.cli run \
    --provider qdrant \
    --dataset benchmarks/inat-gold.json \
    --limit 10

# List available metrics
uv run python -m src.benchmark.cli metrics

# Validate dataset format
uv run python -m src.benchmark.cli validate benchmarks/inat-gold.json
```

## Makefile Targets

```bash
# Run benchmark comparison
make benchmark-compare BENCHMARK_DATASET=benchmarks/inat-gold.json

# Run Qdrant benchmark only
make benchmark-qdrant

# Run Weaviate benchmark only
make benchmark-weaviate

# Validate gold standard dataset
make benchmark-validate BENCHMARK_DATASET=benchmarks/inat-gold.json
```

## Configuration

All settings can be configured via environment variables with the `BENCHMARK_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_K_VALUES` | `[5, 10, 20]` | K values for P@K, R@K, NDCG@K |
| `BENCHMARK_METRICS` | `["precision@k", "recall@k", "map", "ndcg", "mrr"]` | Metrics to compute |
| `BENCHMARK_WARMUP_QUERIES` | `5` | Number of warmup queries before timing |
| `BENCHMARK_COOLDOWN_SECONDS` | `0.1` | Pause between queries |
| `BENCHMARK_OUTPUT_FORMAT` | `console` | Output format: console, json |
| `BENCHMARK_OUTPUT_PATH` | `None` | Path for JSON output file |
| `BENCHMARK_PROVIDERS` | `["qdrant", "weaviate"]` | Providers to benchmark |

## Python API

```python
from src.benchmark.metrics.ir import PrecisionAtK, RecallAtK, NDCG
from src.benchmark.datasets.json_dataset import JSONDataset
from src.benchmark.runner.comparison import ComparisonRunner
from src.benchmark.reporters.console import ConsoleReporter

# Load dataset
dataset = JSONDataset.from_file("benchmarks/inat-gold.json")

# Configure metrics
metrics = [
    PrecisionAtK(k=10),
    RecallAtK(k=10),
    NDCG(k=10),
]

# Run comparison
runner = ComparisonRunner(
    providers={"qdrant": qdrant_client, "weaviate": weaviate_client},
    reporters=[ConsoleReporter()],
)
results = await runner.compare(dataset, metrics)
```

## Integration with Existing Code

- Uses existing `VectorDBProvider` ABC from `src/clients/interfaces/vector_db.py`
- Uses existing `QdrantClient` and `WeaviateClient` implementations
- Uses existing `SearchService` for coordinated searches
- Follows existing patterns for configuration (Pydantic settings)
- Follows existing testing patterns (pytest + AsyncMock)

## Cloud Deployment

The framework is designed for cloud deployment:

- **Stateless**: No local state; all config via environment variables
- **Async**: Non-blocking I/O for high concurrency
- **Configurable**: All settings via `BENCHMARK_*` env vars
- **Containerized**: Works with existing Dockerfile patterns
- **Scalable**: Can run multiple benchmark instances in parallel

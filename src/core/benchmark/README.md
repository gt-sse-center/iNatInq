# Benchmark Framework

Evaluates vector database search quality using standard IR metrics against gold-standard datasets.

## Architecture

```
benchmark/
├── config.py            # BenchmarkConfig (pydantic-settings, BENCHMARK_ prefix)
├── id_mapping.py        # S3KeyIDMapper (doc ID ↔ UUID5 point ID)
├── provider_factory.py  # resolve_search_pipeline() factory
├── search_pipeline.py   # SearchPipeline and QdrantSearchPipeline
├── decorators.py        # @register_metric decorator
├── datasets/            # Dataset loading
│   ├── base.py          # BenchmarkDataset ABC
│   ├── json_dataset.py  # JSONDataset (loads from JSON files)
│   └── schemas.py       # Pydantic models for dataset validation
├── metrics/             # IR metric implementations
│   ├── base.py          # Metric ABC + MetricRegistry
│   ├── builder.py       # build_metrics_from_config()
│   ├── ir.py            # P@K, R@K, MAP, NDCG, MRR
│   └── latency.py       # Query latency metric
├── reporters/           # Output formatters
│   ├── base.py          # Reporter ABC
│   ├── console.py       # ConsoleReporter (stdout table)
│   └── json_reporter.py # JSONReporter (file output)
└── runner/              # Benchmark execution
    ├── base.py          # BenchmarkRunner ABC + BenchmarkResult
    ├── comparison.py    # Multi-provider comparison runner
    └── default.py       # DefaultBenchmarkRunner
```

## Flow

1. **Load dataset** — `JSONDataset.from_file()` parses queries and gold-standard relevance judgments
2. **Resolve provider** — `resolve_search_pipeline()` builds an embedding provider + vector DB pipeline
3. **Run benchmark** — `DefaultBenchmarkRunner.run()` executes queries, collects results
4. **Compute metrics** — Each `Metric.compute()` scores results against gold standard
5. **Report** — `ConsoleReporter` or `JSONReporter` formats output

## Metrics

| Metric | Description |
|--------|-------------|
| `precision@k` | Fraction of top-K results that are relevant |
| `recall@k` | Fraction of relevant documents in top-K |
| `map` | Mean Average Precision (INQUIRE normalization) |
| `ndcg` | Normalized Discounted Cumulative Gain |
| `mrr` | Mean Reciprocal Rank |

## Datasets

Datasets follow the JSON schema in `benchmarks/schemas/gold-standard.schema.json`. Each dataset contains queries with relevance judgments mapping query text to relevant document IDs.

## Configuration

All settings use the `BENCHMARK_` env var prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_K_VALUES` | `[50]` | K values for top-K metrics |
| `BENCHMARK_METRICS` | all five | Metrics to compute |
| `BENCHMARK_WARMUP_QUERIES` | `5` | Warmup queries before measurement |
| `BENCHMARK_OUTPUT_FORMAT` | `console` | `console`, `json`, or `both` |

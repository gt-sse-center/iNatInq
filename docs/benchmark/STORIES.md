# Benchmarking Framework - Agile Stories

Stories for implementing the Vector DB Benchmarking Framework. All stories are sized at 5 points or fewer.

## Development Guidelines

> **Unit Tests**: Written as part of each story. Every story's acceptance criteria includes unit test coverage for the implemented functionality.
>
> **Makefile Targets**: Added incrementally as needed when a story introduces CLI commands or commonly-run operations.
>
> **Integration Testing**: Should begin as early as possible once core components are in place. The integration test story can run in parallel with later stories.

---

## Epic: Benchmark Metrics

### Story 1: Metric ABC and MetricRegistry Metaclass

**Points:** 3

**Description:**
Create the foundational abstraction for benchmark metrics using an Abstract Base Class with a metaclass for auto-registration.

**Technical Notes:**
- Create `src/benchmark/metrics/base.py`
- Implement `MetricRegistry` metaclass with `_registry` dict
- Metaclass auto-registers subclasses that have a `name` class variable
- Implement `Metric` ABC with `compute()` abstract method
- Signature: `compute(retrieved: Sequence[str], relevant: set[str], graded: dict[str, int] | None) -> float`
- Include `get()` and `all_metrics()` class methods on registry

**Acceptance Criteria:**
- [ ] `MetricRegistry` metaclass auto-registers subclasses with `name` attribute
- [ ] `Metric` ABC defines `name`, `description` class variables and `compute()` abstract method
- [ ] `MetricRegistry.get("metric_name")` returns the metric class
- [ ] `MetricRegistry.all_metrics()` returns list of registered metric names
- [ ] Unit tests verify registration and retrieval

---

### Story 2: Precision@K and Recall@K Metrics

**Points:** 3

**Description:**
Implement Precision@K and Recall@K information retrieval metrics.

**Technical Notes:**
- Create `src/benchmark/metrics/ir.py`
- Use `@attrs.define` for both classes
- `PrecisionAtK`: `len(set(top_k) & relevant) / k`
- `RecallAtK`: `len(set(top_k) & relevant) / len(relevant)`
- Handle edge cases: k=0, empty relevant set
- Both take `k: int = 10` parameter

**Acceptance Criteria:**
- [ ] `PrecisionAtK(k=10).compute(retrieved, relevant)` returns correct precision
- [ ] `RecallAtK(k=10).compute(retrieved, relevant)` returns correct recall
- [ ] Edge cases return 0.0 (not errors)
- [ ] Metrics auto-register with MetricRegistry
- [ ] Unit tests with known inputs/outputs

---

### Story 3: MAP and MRR Metrics

**Points:** 3

**Description:**
Implement Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR) metrics.

**Technical Notes:**
- Add to `src/benchmark/metrics/ir.py`
- MAP: Sum of (precision at each relevant hit) / total relevant docs
- MRR: 1/rank of first relevant result, 0 if none found
- Both use `@attrs.define`

**Acceptance Criteria:**
- [ ] `MeanAveragePrecision().compute()` calculates correct MAP
- [ ] `MRR().compute()` returns reciprocal of first relevant doc rank
- [ ] Returns 0.0 when no relevant docs in results
- [ ] Metrics auto-register with MetricRegistry
- [ ] Unit tests with known inputs/outputs

---

### Story 4: NDCG Metric

**Points:** 3

**Description:**
Implement Normalized Discounted Cumulative Gain (NDCG) for graded relevance scoring.

**Technical Notes:**
- Add to `src/benchmark/metrics/ir.py`
- DCG = sum of relevance[i] / log2(i+2) for top-k
- IDCG = DCG with ideal ordering (sorted relevance scores)
- NDCG = DCG / IDCG
- Falls back to binary relevance if `graded` not provided
- Takes `k: int = 10` parameter

**Acceptance Criteria:**
- [ ] `NDCG(k=10).compute(retrieved, relevant, graded)` calculates correct NDCG
- [ ] Works with binary relevance (graded=None)
- [ ] Works with graded relevance (graded={doc: score})
- [ ] Returns 0.0 when IDCG is 0
- [ ] Unit tests with known inputs/outputs

---

### Story 5: Latency Statistics

**Points:** 2

**Description:**
Implement latency statistics collection and percentile calculations.

**Technical Notes:**
- Create `src/benchmark/metrics/latency.py`
- Use `@attrs.define` with `samples: list[float]`
- Properties: `p50`, `p95`, `p99`, `mean`, `qps`
- Use `numpy.percentile()` for calculations
- `to_dict()` returns all stats with `_ms` suffix for latencies

**Acceptance Criteria:**
- [ ] `LatencyStats` collects latency samples
- [ ] `p50`, `p95`, `p99` properties return correct percentiles
- [ ] `qps` returns queries per second (count / total time)
- [ ] `to_dict()` returns formatted dict with millisecond values
- [ ] Handles empty samples list gracefully (returns 0.0)
- [ ] Unit tests verify calculations

---

## Epic: Dataset Loading

### Story 6: Dataset ABC and Query Model

**Points:** 2

**Description:**
Create abstract base class for benchmark datasets and the Query data model.

**Technical Notes:**
- Create `src/benchmark/datasets/base.py`
- `Query` class with attrs: `id`, `text`, `relevant: set[str]`, `graded_relevance: dict[str, int]`
- `Dataset` ABC with: `name` property, `modality` property, `queries()` iterator, `__len__()`

**Acceptance Criteria:**
- [ ] `Query` attrs class holds query data with relevant doc IDs
- [ ] `Dataset` ABC defines required interface
- [ ] `modality` returns "text" or "image"
- [ ] `queries()` returns `Iterator[Query]`
- [ ] `__len__()` returns query count

---

### Story 7: Pydantic Schemas for Dataset Validation

**Points:** 3

**Description:**
Create Pydantic models for validating gold standard dataset JSON files.

**Technical Notes:**
- Create `src/benchmark/datasets/schemas.py`
- `QuerySchema`: id, text, relevant (list), graded_relevance (optional dict)
- `MetadataSchema`: created_at, annotators, agreement_threshold (all optional)
- `DatasetSchema`: name, description, modality, queries (list), metadata
- Use Pydantic v2 model_validator for custom validation

**Acceptance Criteria:**
- [ ] `DatasetSchema.model_validate(json_dict)` validates dataset structure
- [ ] Validation fails on missing required fields
- [ ] Validation fails on invalid modality (not "text" or "image")
- [ ] `graded_relevance` is optional
- [ ] Unit tests for valid and invalid inputs

---

### Story 8: JSON Dataset Loader

**Points:** 3

**Description:**
Implement JSON file loader for gold standard datasets.

**Technical Notes:**
- Create `src/benchmark/datasets/json_dataset.py`
- `JSONDataset(Dataset)` implementation
- Class method `from_file(path: Path) -> JSONDataset`
- Validates with Pydantic schema on load
- Caches queries in memory after loading

**Acceptance Criteria:**
- [ ] `JSONDataset.from_file("path.json")` loads and validates dataset
- [ ] Raises clear error on invalid JSON or schema
- [ ] `queries()` iterates over `Query` objects
- [ ] `name` and `modality` properties return correct values
- [ ] Unit tests with sample dataset file

---

## Epic: Benchmark Runner

### Story 9: BenchmarkRunner ABC and BenchmarkResult

**Points:** 3

**Description:**
Create abstract base class for benchmark runners and result data model.

**Technical Notes:**
- Create `src/benchmark/runner/base.py`
- `BenchmarkResult` attrs: provider, dataset, metrics dict, latency dict, timestamp, config
- `BenchmarkRunner` ABC with `run()` method
- Signature: `async run(provider, dataset, metrics, *, limit=10, warmup_queries=5) -> BenchmarkResult`

**Acceptance Criteria:**
- [ ] `BenchmarkResult` holds all benchmark output data
- [ ] `BenchmarkRunner` ABC defines async `run()` interface
- [ ] `timestamp` defaults to current UTC time
- [ ] Type hints use existing `VectorDBProvider` interface

---

### Story 10: Default Benchmark Runner Implementation

**Points:** 5

**Description:**
Implement the default benchmark runner that executes queries and computes metrics.

**Technical Notes:**
- Add `DefaultBenchmarkRunner(BenchmarkRunner)` to `base.py`
- Warmup phase: run N queries without timing
- Timing phase: measure latency for each query
- Compute all metrics for each query, aggregate (mean)
- Collect latency samples into `LatencyStats`
- Handle async search via `provider.search_images()` or `provider.search()`

**Acceptance Criteria:**
- [ ] Runs warmup queries before timing
- [ ] Measures latency per query during benchmark
- [ ] Computes each metric and returns mean across queries
- [ ] Returns `BenchmarkResult` with metrics and latency dicts
- [ ] Works with both text and image modalities
- [ ] Unit tests with mock provider

---

### Story 11: Comparison Runner

**Points:** 3

**Description:**
Implement runner that benchmarks multiple providers for comparison.

**Technical Notes:**
- Create `src/benchmark/runner/comparison.py`
- `ComparisonRunner` with `providers: dict[str, VectorDBProvider]` and `reporters: list[Reporter]`
- `compare()` method runs benchmark on each provider
- Calls reporters after all benchmarks complete
- Default metrics if none specified

**Acceptance Criteria:**
- [ ] `ComparisonRunner.compare(dataset)` benchmarks all providers
- [ ] Returns `dict[str, BenchmarkResult]` keyed by provider name
- [ ] Calls all reporters with results
- [ ] Uses default metrics (P@K, R@K, MAP, NDCG, MRR) if none specified
- [ ] Unit tests with mock providers

---

## Epic: Reporters

### Story 12: Reporter ABC

**Points:** 1

**Description:**
Create abstract base class for benchmark result reporters.

**Technical Notes:**
- Create `src/benchmark/reporters/base.py`
- `Reporter` ABC with `async report(results: dict[str, BenchmarkResult]) -> None`

**Acceptance Criteria:**
- [ ] `Reporter` ABC defines `report()` interface
- [ ] Accepts dict of provider name to BenchmarkResult
- [ ] Is async for flexibility

---

### Story 13: Console Reporter

**Points:** 3

**Description:**
Implement reporter that outputs formatted results to console.

**Technical Notes:**
- Create `src/benchmark/reporters/console.py`
- Pretty table format with provider comparison
- Show metrics side-by-side
- Show latency stats (p50, p95, p99, QPS)
- Use rich library if available, fall back to plain text

**Acceptance Criteria:**
- [ ] Prints formatted comparison table to stdout
- [ ] Shows all metrics for each provider
- [ ] Shows latency percentiles and QPS
- [ ] Handles single provider (no comparison)
- [ ] Output is readable and well-aligned

---

### Story 14: JSON Reporter

**Points:** 2

**Description:**
Implement reporter that outputs results to JSON file.

**Technical Notes:**
- Create `src/benchmark/reporters/json_reporter.py`
- `JSONReporter(output_path: Path)`
- Serialize `BenchmarkResult` to JSON
- Include timestamp, config, all metrics and latency

**Acceptance Criteria:**
- [ ] Writes valid JSON to specified file path
- [ ] JSON includes all result data (metrics, latency, timestamp, config)
- [ ] Handles datetime serialization
- [ ] Creates parent directories if needed
- [ ] Unit tests verify JSON structure

---

## Epic: Infrastructure

### Story 15: Timing Decorators

**Points:** 2

**Description:**
Implement decorators for timing and benchmarking.

**Technical Notes:**
- Create `src/benchmark/decorators.py`
- `@timed`: Returns `(result, elapsed_seconds)` tuple
- `@benchmark_provider(warmup=5, cooldown=0.1)`: Runs warmup before measurement
- Both support async functions

**Acceptance Criteria:**
- [ ] `@timed` returns result and elapsed time
- [ ] `@benchmark_provider` runs warmup queries first
- [ ] Cooldown pause between warmup queries
- [ ] Works with async functions
- [ ] Unit tests verify timing accuracy

---

### Story 16: Benchmark Configuration

**Points:** 3

**Description:**
Implement Pydantic settings for benchmark configuration.

**Technical Notes:**
- Create `src/benchmark/config.py`
- `BenchmarkConfig(BaseSettings)` with `env_prefix="BENCHMARK_"`
- Settings: k_values, metrics, warmup_queries, cooldown_seconds, concurrent_queries
- Settings: output_format, output_path, providers

**Acceptance Criteria:**
- [ ] Config loads from environment variables
- [ ] `BENCHMARK_K_VALUES` parses as list of ints
- [ ] `BENCHMARK_PROVIDERS` parses as list of strings
- [ ] All settings have sensible defaults
- [ ] Unit tests verify env var loading

---

### Story 17: CLI Implementation

**Points:** 5

**Description:**
Implement Click CLI for running benchmarks from command line.

**Technical Notes:**
- Create `src/benchmark/cli.py`
- Commands: `compare`, `run`, `metrics`, `validate`
- Use existing provider factory to instantiate clients
- Wire up reporters based on `--output` format
- Load config from environment, override with CLI args

**Acceptance Criteria:**
- [ ] `python -m src.benchmark.cli compare --dataset X --providers Y` runs comparison
- [ ] `python -m src.benchmark.cli run --provider X` runs single benchmark
- [ ] `python -m src.benchmark.cli metrics` lists available metrics
- [ ] `python -m src.benchmark.cli validate FILE` validates dataset JSON
- [ ] CLI args override environment config
- [ ] Helpful error messages on invalid input

---

## Epic: Testing and Documentation

### Story 18: Sample Gold Standard Dataset

**Points:** 2

**Description:**
Create sample gold standard dataset for testing and documentation.

**Technical Notes:**
- Create `benchmarks/sample/sample-gold.json`
- Use synthetic image data (red circle, blue square, etc.)
- 10-20 queries with known relevant docs
- Include both binary and graded relevance examples
- Create JSON Schema at `benchmarks/schemas/gold-standard.schema.json`

**Acceptance Criteria:**
- [ ] Sample dataset passes schema validation
- [ ] Queries reference synthetic image filenames
- [ ] Mix of easy queries (single color) and hard queries (color + shape)
- [ ] JSON Schema documents all fields
- [ ] README explains how to create gold standard datasets

---

### Story 19: Integration Test with Synthetic Data

**Points:** 5

**Description:**
Create end-to-end integration test using synthetic images and gold standard dataset. **This story should begin as early as possible** once core components (metrics, dataset loader, runner) are in place.

**Technical Notes:**
- Create `tests/integration/test_benchmark_e2e.py`
- Generate synthetic images, upload to MinIO, index in vector DBs
- Create gold standard based on known synthetic image properties
- Run full benchmark comparison
- Verify metrics are reasonable (not 0.0 or 1.0 for all)

**Acceptance Criteria:**
- [ ] Test generates and indexes synthetic images
- [ ] Gold standard queries match synthetic image properties (e.g., "red circle")
- [ ] Benchmark runs against both Qdrant and Weaviate
- [ ] Precision/Recall values are reasonable for known relevance
- [ ] Test cleans up after itself
- [ ] Can run via `make test-integration`

---

## Story Summary

| Epic | Story | Points |
|------|-------|--------|
| Metrics | Metric ABC and MetricRegistry | 3 |
| Metrics | Precision@K and Recall@K | 3 |
| Metrics | MAP and MRR | 3 |
| Metrics | NDCG | 3 |
| Metrics | Latency Statistics | 2 |
| Datasets | Dataset ABC and Query Model | 2 |
| Datasets | Pydantic Schemas | 3 |
| Datasets | JSON Dataset Loader | 3 |
| Runner | BenchmarkRunner ABC | 3 |
| Runner | Default Benchmark Runner | 5 |
| Runner | Comparison Runner | 3 |
| Reporters | Reporter ABC | 1 |
| Reporters | Console Reporter | 3 |
| Reporters | JSON Reporter | 2 |
| Infrastructure | Timing Decorators | 2 |
| Infrastructure | Benchmark Configuration | 3 |
| Infrastructure | CLI Implementation | 5 |
| Testing | Sample Gold Standard Dataset | 2 |
| Testing | Integration Test | 5 |
| **Total** | | **57** |

## Suggested Sprint Breakdown

**Sprint 1 - Core Metrics (14 points)**
- Story 1: Metric ABC and MetricRegistry (3)
- Story 2: Precision@K and Recall@K (3)
- Story 3: MAP and MRR (3)
- Story 4: NDCG (3)
- Story 5: Latency Statistics (2)

**Sprint 2 - Data and Runner (16 points)**
- Story 6: Dataset ABC and Query Model (2)
- Story 7: Pydantic Schemas (3)
- Story 8: JSON Dataset Loader (3)
- Story 9: BenchmarkRunner ABC (3)
- Story 10: Default Benchmark Runner (5)

**Sprint 3 - Comparison, Reporters, and Early Integration (14 points)**
- Story 11: Comparison Runner (3)
- Story 12: Reporter ABC (1)
- Story 13: Console Reporter (3)
- Story 14: JSON Reporter (2)
- Story 19: Integration Test (5) - *Start early to validate end-to-end flow*

**Sprint 4 - CLI and Infrastructure (12 points)**
- Story 15: Timing Decorators (2)
- Story 16: Benchmark Configuration (3)
- Story 17: CLI Implementation (5)
- Story 18: Sample Gold Standard Dataset (2)

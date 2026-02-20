# Revise Benchmark Integration Tests with Real INQUIRE Images

## Context

The current `tests/integration/benchmark/test_benchmark_e2e.py` (branch `djalali/956_integration_test`) uses in-memory synthetic data with mock providers. We want to revise these tests to use **real iNat images** from the INQUIRE benchmark, exercise the **CLI**, and reference datasets from the top-level `benchmarks/` directory.

**Selected fixture query**: Query 127 — "a peach-faced Lovebird with the turquoise mutation" (52 relevant images). Images live in the remote MinIO (`http://20.119.101.101:9000`, bucket `inquire-train-data`, flat keys = image_id strings).

**Approach**: Download 52 images once, commit them as fixtures under `syntheticdata/data/inquire/query_127/`, and load them into the local testcontainer MinIO during integration tests.

## Tasks

### 1. Create download script

**New file**: `syntheticdata/download_inquire_fixtures.py`

- Uses `boto3` directly (consistent with `syntheticdata/synthetic_data.py` patterns)
- Hardcodes the 52 image IDs for query 127
- Downloads from `inquire-train-data` bucket on remote MinIO
- Saves to `syntheticdata/data/inquire/query_127/{image_id}.jpg`
- CLI via `argparse`: `--endpoint`, `--dry-run`, skip already-downloaded files
- No pandas, no app-layer dependencies

### 2. Run download script, update .gitignore

- Run script to populate `syntheticdata/data/inquire/query_127/` (~52 JPEGs, ~5MB total)
- Update `syntheticdata/.gitignore` to add negation rule so `data/inquire/` is tracked:
  ```
  !data/inquire/
  ```

### 3. Create benchmark integration conftest

**New file**: `tests/integration/benchmark/conftest.py`

Path constants pointing to `benchmarks/` directory and fixture images:
- `INQUIRE_FIXTURES_DIR` → `syntheticdata/data/inquire/query_127/`
- `INQUIRE_VAL_PATH` → `benchmarks/inquire/inquire-val.json`
- `SAMPLE_GOLD_PATH` → `benchmarks/sample/sample-gold.json`

Fixtures:
- `inquire_images_dir` (session) — validates fixtures exist, `pytest.skip()` if missing
- `inquire_bucket` (session) — uses shared `minio_client` from `tests/integration/clients/conftest.py`, creates `inquire-train-data` bucket, uploads all 52 images once
- `inquire_val_dataset` (session) — `JSONDataset.from_file(INQUIRE_VAL_PATH)`
- `sample_gold_dataset` (session) — `JSONDataset.from_file(SAMPLE_GOLD_PATH)`
- `cli_runner` — `typer.testing.CliRunner()`

### 4. Revise test_benchmark_e2e.py

**Keep** existing `TestBenchmarkE2E` class (8 mock-based async tests — these test runner/reporter pipeline logic).

**Add** three new test classes:

**`TestCLIWithRealDatasets`** — exercises CLI `validate` and `metrics` commands against real `benchmarks/` files:
- `test_validate_inquire_val` — validates `inquire-val.json`, checks name/modality/query count in output
- `test_validate_inquire_val_modality` — confirms `image` modality in output
- `test_validate_sample_gold` — validates `sample-gold.json`
- `test_metrics_command` — lists all registered IR metrics (precision@k, recall@k, map, ndcg, mrr)

**`TestInquireDatasetIntegration`** — loads real INQUIRE benchmark files:
- `test_inquire_val_loads` — name and modality correct
- `test_inquire_val_query_count` — 50 queries
- `test_query_127_present` — query 127 exists with 52 relevant images, text contains "peach-faced Lovebird"
- `test_relevant_ids_are_numeric_strings` — all relevant IDs are digit strings

**`TestInquireImageFixtures`** — validates fixture images and MinIO testcontainer integration:
- `test_fixture_images_exist` — 52 JPEGs on disk
- `test_fixture_images_are_valid` — non-empty, JPEG magic bytes
- `test_images_loaded_in_minio` — 52 objects in testcontainer bucket
- `test_image_roundtrip` — image from MinIO matches local fixture
- `test_query_127_images_all_present` — all 52 relevant IDs for query 127 are in MinIO

## Files

| File | Action | Purpose |
|------|--------|---------|
| `syntheticdata/download_inquire_fixtures.py` | Create | Download script for 52 INQUIRE images |
| `syntheticdata/.gitignore` | Edit | Add `!data/inquire/` negation rule |
| `syntheticdata/data/inquire/query_127/*.jpg` | Create (52 files) | Committed image fixtures |
| `tests/integration/benchmark/conftest.py` | Create | Fixtures for datasets, CLI runner, MinIO image loading |
| `tests/integration/benchmark/test_benchmark_e2e.py` | Edit | Add 3 new test classes alongside existing mock tests |

No changes to schema, domain models, CLI, or existing client/ingestion code.

## Verification

```bash
# Download fixtures (one-time)
uv run python syntheticdata/download_inquire_fixtures.py

# Run benchmark integration tests
uv run pytest tests/integration/benchmark/ -v

# Existing unit tests still pass
uv run pytest tests/unit/core/benchmark/ -v --no-cov

# Lint
uv run ruff check syntheticdata/ tests/integration/benchmark/
uv run ruff format --check syntheticdata/ tests/integration/benchmark/
```

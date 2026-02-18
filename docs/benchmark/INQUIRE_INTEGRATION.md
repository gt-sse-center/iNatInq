# Integrate INQUIRE Dataset into Benchmark Framework

## Context

The current `benchmarks/sample/sample-gold.json` contains 10 synthetic queries with fake document IDs. The real gold standard should be based on the [INQUIRE benchmark](https://github.com/inquire-benchmark/INQUIRE) — a NeurIPS 2024 expert-level text-to-image retrieval benchmark over iNaturalist 2024 (5M images, 250 queries, ~33k binary relevance annotations).

**INQUIRE ships 3 CSVs** (in `data/inquire/`):

| File | Rows | Columns |
|------|------|---------|
| `inquire_queries_val.csv` | 50 | `query_id, query_text, supercategory, category, iconic_group` |
| `inquire_queries_test.csv` | 200 | same |
| `inquire_annotations.csv` | ~32,700 | `query_id, image_id, image_path` |

Annotations are binary — each row is a (query, relevant image) pair. No graded relevance.

## Decisions

- **No schema changes** — drop INQUIRE query metadata (supercategory, category, iconic_group)
- **Both splits** — generate `inquire-val.json` (50 queries) and `inquire-test.json` (200 queries)
- **Document IDs** — `image_id` stringified (canonical iNat24 identifier)
- **Modality** — `"image"` (text-to-image retrieval)
- **No graded relevance** — INQUIRE is binary; `graded_relevance` omitted
- **Keep existing sample** — `benchmarks/sample/sample-gold.json` stays as lightweight test fixture

## Tasks

### 1. Create conversion script

**File**: `scripts/convert_inquire.py`

- Downloads the 3 CSVs from the INQUIRE GitHub repo via raw URL
- Parses with `csv` module (no pandas dependency)
- Joins queries with annotations by `query_id`
- Outputs two JSON files matching our `gold-standard.schema.json`
- Uses `image_id` (as str) for `relevant` arrays
- CLI: `python scripts/convert_inquire.py --output-dir benchmarks/inquire/`

### 2. Generate dataset files

Run the script to produce:

- `benchmarks/inquire/inquire-val.json` — 50 queries, modality `"image"`
- `benchmarks/inquire/inquire-test.json` — 200 queries, modality `"image"`

Output JSON structure:

```json
{
  "name": "inquire-val",
  "description": "INQUIRE validation split — 50 expert text-to-image queries over iNat24.",
  "modality": "image",
  "queries": [
    {
      "id": "109",
      "text": "Eurasian Black Grouse male",
      "relevant": ["4702270", "2918841", ...]
    }
  ],
  "metadata": {
    "created_at": "2024-10-08",
    "annotators": ["inquire-benchmark"]
  }
}
```

### 3. Add unit tests for INQUIRE datasets

**File**: `tests/unit/core/benchmark/test_inquire_dataset.py`

- Loads each generated JSON via `JSONDataset.from_file()`
- Asserts query counts (50 val, 200 test)
- Asserts modality is `"image"`
- Asserts every query has at least 1 relevant doc
- Asserts all query IDs are unique
- Asserts all relevant IDs are non-empty strings

### 4. Existing tests unchanged

`tests/unit/core/benchmark/test_sample_dataset.py` stays as-is.

## Files Created

| File | Purpose |
|------|---------|
| `docs/benchmark/INQUIRE_INTEGRATION.md` | This plan |
| `scripts/convert_inquire.py` | CSV → JSON conversion script |
| `benchmarks/inquire/inquire-val.json` | 50-query validation split |
| `benchmarks/inquire/inquire-test.json` | 200-query test split |
| `tests/unit/core/benchmark/test_inquire_dataset.py` | Dataset validation tests |

No changes to schema, domain models, or existing code.

## Verification

```bash
python scripts/convert_inquire.py --output-dir benchmarks/inquire/
uv run pytest tests/unit/core/benchmark/test_inquire_dataset.py -v
uv run pytest tests/unit/core/benchmark/test_sample_dataset.py -v
uv run ruff check scripts/ src/ tests/
```

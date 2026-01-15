# Synthetic Test Data

Tools for generating and uploading synthetic test data for the ML pipeline.

## Overview

This module provides:

- **DocumentGenerator**: Splits source text into chunks for testing
- **MinIOUploader**: Uploads documents to S3/MinIO with concurrent workers
- **CLI**: Command-line interface for generation, upload, and setup

## Design

1. **Deterministic Output**: Same input produces identical documents
2. **Sentence-Aware Chunking**: Breaks at sentence boundaries when possible
3. **Concurrent Uploads**: Uses thread pool for fast uploads (default: 50 workers)
4. **Retry Logic**: Exponential backoff for transient failures

## Structure

```text
syntheticdata/
├── synthetic_data.py   # Main script (DocumentGenerator + MinIOUploader)
├── moby-dick.txt       # Source text (Moby Dick from Project Gutenberg)
├── inputs/             # Generated test documents (gitignored)
└── README.md           # This file
```

## Quick Start

### Generate and Upload (Recommended)

```bash
make syntheticdata-setup COUNT=1000
```

### Step by Step

```bash
# Generate documents
make syntheticdata-generate COUNT=1000

# Upload to MinIO
make syntheticdata-upload
```

### Clean Up

```bash
make syntheticdata-clean
```

## CLI Usage

The `synthetic_data.py` script provides three commands:

### Generate Documents

```bash
python3 syntheticdata/synthetic_data.py generate --count 1000 --chunk-size 500
```

Options:

- `--count`: Number of documents to generate (default: 1000)
- `--chunk-size`: Target characters per chunk (default: 500)
- `--source`: Source text file (default: moby-dick.txt)
- `--output`: Output directory (default: inputs/)

### Upload to MinIO

```bash
python3 syntheticdata/synthetic_data.py upload --endpoint http://localhost:9000
```

Options:

- `--endpoint`: MinIO endpoint URL (default: <http://localhost:9000>)
- `--bucket`: Target bucket (default: pipeline)
- `--prefix`: S3 prefix (default: inputs/)
- `--max-workers`: Concurrent uploads (default: 50)

### All-in-One Setup

```bash
python3 syntheticdata/synthetic_data.py setup --count 1000 --endpoint http://localhost:9000
```

## Classes

### DocumentGenerator

Generates test documents from a source text file:

```python
from syntheticdata.synthetic_data import DocumentGenerator

generator = DocumentGenerator(
    source_file="moby-dick.txt",
    output_dir="inputs",
    chunk_size=500,
)
generator.generate(count=1000)
```

### MinIOUploader

Uploads files to MinIO with concurrent uploads and retry logic:

```python
from syntheticdata.synthetic_data import MinIOUploader

uploader = MinIOUploader(
    endpoint="http://localhost:9000",
    bucket="pipeline",
    max_workers=50,
)
uploader.upload_directory(input_dir="inputs", prefix="inputs/")
```

## Workflow

1. **Generate documents** from source text:

   ```bash
   make syntheticdata-generate COUNT=1000
   ```

2. **Upload to MinIO**:

   ```bash
   make syntheticdata-upload
   ```

3. **Process with Ray job**:

   ```bash
   curl -X POST http://localhost:8000/ray/jobs \
     -H "Content-Type: application/json" \
     -d '{"s3_prefix": "inputs/", "collection": "documents"}'
   ```

4. **Search the indexed documents**:

   ```bash
   curl "http://localhost:8000/search?q=whale&limit=10"
   ```

## Source Material

The test data is generated from **Moby Dick** by Herman Melville:

- **Public Domain**: Published in 1851
- **Size**: ~1.2 million characters
- **Source**: [Project Gutenberg](https://www.gutenberg.org/)

# Synthetic Test Data

Tools for generating and uploading synthetic test data for the ML pipeline.

## Overview

This module provides:

- **TextGenerator**: Splits source text into chunks for testing
- **ImageGenerator**: Creates images with semantic content (shapes + colors)
- **MinIOUploader**: Uploads files to S3/MinIO with concurrent workers
- **CLI**: Command-line interface for generation, upload, and setup

## Directory Structure

```text
syntheticdata/
├── synthetic_data.py      # Main script (TextGenerator + ImageGenerator + MinIOUploader)
├── .gitignore             # Ignores generated data directories
├── README.md              # This file
├── data/
│   ├── imgs/              # Generated images (gitignored)
│   └── txts/              # Generated text chunks (gitignored)
└── seed/
    ├── img/               # Base images for reference
    │   ├── base-circle.png
    │   ├── base-square.png
    │   └── base-triangle.png
    └── txt/
        └── moby-dick.txt  # Seed text file (Moby Dick from Project Gutenberg)
```

## Quick Start

### Text Pipeline

```bash
# Generate and upload text in one step
make synthetic-text-setup COUNT=1000

# Or step by step
make synthetic-text-generate COUNT=1000
make synthetic-text-upload
```

### Image Pipeline

```bash
# Generate and upload images in one step
make synthetic-images-setup IMAGE_COUNT=100

# Or step by step
make synthetic-images-generate IMAGE_COUNT=100
make synthetic-images-upload
```

### Both Pipelines

```bash
# Generate and upload both text and images
uv run python syntheticdata/synthetic_data.py setup-all --count 1000 --image-count 100
```

### Clean Up

```bash
make synthetic-text-clean    # Remove generated text
make synthetic-images-clean  # Remove generated images
make synthetic-clean-all     # Remove both
```

## CLI Usage

The `synthetic_data.py` script provides multiple commands:

### Text Commands

```bash
# Generate text documents
uv run python syntheticdata/synthetic_data.py generate-text --count 1000 --chunk-size 500

# Upload text documents
uv run python syntheticdata/synthetic_data.py upload-text --endpoint http://localhost:9000

# Generate and upload text
uv run python syntheticdata/synthetic_data.py setup-text --count 1000
```

Options for text commands:

- `--count`: Number of documents to generate (default: 1000)
- `--chunk-size`: Target characters per chunk (default: 500)
- `--source`: Source text file (default: seed/txt/moby-dick.txt)
- `--output`: Output directory (default: data/txts/)
- `--prefix`: S3 prefix (default: inputs/)

### Image Commands

```bash
# Generate images
uv run python syntheticdata/synthetic_data.py generate-images --count 100 --size 512

# Upload images
uv run python syntheticdata/synthetic_data.py upload-images --endpoint http://localhost:9000

# Generate and upload images
uv run python syntheticdata/synthetic_data.py setup-images --count 100
```

Options for image commands:

- `--count`: Number of images to generate (default: 100)
- `--size`: Image size in pixels (default: 512)
- `--seed`: Random seed for determinism (default: 42)
- `--output`: Output directory (default: data/imgs/)
- `--prefix`: S3 prefix (default: images/)

### Combined Command

```bash
# Generate and upload both text and images
uv run python syntheticdata/synthetic_data.py setup-all \
    --count 1000 \
    --image-count 100 \
    --endpoint http://localhost:9000
```

### Common MinIO Options

All upload commands support:

- `--endpoint`: MinIO endpoint URL (default: http://localhost:9000)
- `--bucket`: Target bucket (default: pipeline)
- `--max-workers`: Concurrent uploads (default: 50)
- `--access-key`: MinIO access key (default: minioadmin)
- `--secret-key`: MinIO secret key (default: minioadmin)

## Classes

### TextGenerator

Generates test documents from a source text file:

```python
from syntheticdata.synthetic_data import TextGenerator

generator = TextGenerator(
    source_file="seed/txt/moby-dick.txt",
    output_dir="data/txts",
    chunk_size=500,
)
generator.generate(count=1000)
```

### ImageGenerator

Generates synthetic test images:

```python
from syntheticdata.synthetic_data import ImageGenerator

generator = ImageGenerator(
    output_dir="data/imgs",
    size=512,
    seed=42,
)
generator.generate(count=100)
```

Images include:

- 8 colors: red, green, blue, yellow, orange, purple, pink, teal
- 3 shapes: circle, square, triangle
- 2 backgrounds: solid, gradient

Filenames encode content: `{color}-{shape}-{background}-{index}.png`

### MinIOUploader

Uploads files to MinIO with concurrent uploads and retry logic:

```python
from syntheticdata.synthetic_data import MinIOUploader

uploader = MinIOUploader(
    endpoint="http://localhost:9000",
    bucket="pipeline",
    max_workers=50,
)

# Upload text files
uploader.upload_directory(input_dir="data/txts", prefix="inputs/", pattern="*.txt")

# Upload images
uploader.upload_directory(input_dir="data/imgs", prefix="images/", pattern="*.png")
```

## Design

1. **Deterministic Output**: Same input produces identical documents/images
2. **Sentence-Aware Chunking**: Text breaks at sentence boundaries when possible
3. **Semantic Filenames**: Image filenames encode visual content
4. **Concurrent Uploads**: Uses thread pool for fast uploads (default: 50 workers)
5. **Retry Logic**: Exponential backoff for transient failures
6. **Unified Uploader**: Single class handles both text and images with auto content-type detection

## Workflow

1. **Generate documents** from source text:

   ```bash
   make synthetic-text-generate COUNT=1000
   ```

2. **Upload to MinIO**:

   ```bash
   make synthetic-text-upload
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

### Text

The test data is generated from **Moby Dick** by Herman Melville:

- **Public Domain**: Published in 1851
- **Size**: ~1.2 million characters
- **Source**: [Project Gutenberg](https://www.gutenberg.org/)

### Images

Base reference images in `seed/img/`:

- `base-circle.png` - Red circle on white background
- `base-square.png` - Blue square on white background
- `base-triangle.png` - Green triangle on white background

Generated images use these shapes with randomized colors and backgrounds.

# Synthetic Test Data

Tools for generating and uploading synthetic test images for the ML pipeline.

## Overview

This module provides:

- **ImageGenerator**: Creates images with semantic content (shapes + colors)
- **MinIOUploader**: Uploads files to S3/MinIO with concurrent workers
- **CLI**: Command-line interface for generation, upload, and setup

## Directory Structure

```text
syntheticdata/
├── synthetic_data.py      # Main script (ImageGenerator + MinIOUploader)
├── .gitignore             # Ignores generated data directories
├── README.md              # This file
├── data/
│   └── imgs/              # Generated images (gitignored)
└── seed/
    └── img/               # Base images for reference
        ├── base-circle.png
        ├── base-square.png
        └── base-triangle.png
```

## Quick Start

```bash
# Generate and upload images in one step
make synthetic-images-setup IMAGE_COUNT=100

# Or step by step
make synthetic-images-generate IMAGE_COUNT=100
make synthetic-images-upload
```

### Clean Up

```bash
make synthetic-images-clean  # Remove generated images
```

## CLI Usage

The `synthetic_data.py` script provides multiple commands:

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

### Common MinIO Options

All upload commands support:

- `--endpoint`: MinIO endpoint URL (default: http://localhost:9000)
- `--bucket`: Target bucket (default: pipeline)
- `--max-workers`: Concurrent uploads (default: 50)
- `--access-key`: MinIO access key (default: minioadmin)
- `--secret-key`: MinIO secret key (default: minioadmin)

## Classes

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

# Upload images
uploader.upload_directory(input_dir="data/imgs", prefix="images/", pattern="*.png")
```

## Design

1. **Deterministic Output**: Same input produces identical images
2. **Semantic Filenames**: Image filenames encode visual content
3. **Concurrent Uploads**: Uses thread pool for fast uploads (default: 50 workers)
4. **Retry Logic**: Exponential backoff for transient failures

## Workflow

1. **Generate images**:

   ```bash
   make synthetic-images-generate IMAGE_COUNT=100
   ```

2. **Upload to MinIO**:

   ```bash
   make synthetic-images-upload
   ```

3. **Process with Ray job**:

   ```bash
   curl -X POST http://localhost:8000/ray/jobs \
     -H "Content-Type: application/json" \
     -d '{"s3_prefix": "images/", "collection": "documents"}'
   ```

4. **Search the indexed images**:

   ```bash
   curl "http://localhost:8000/search/images?q=red+circle&limit=10"
   ```

## Source Material

Base reference images in `seed/img/`:

- `base-circle.png` - Red circle on white background
- `base-square.png` - Blue square on white background
- `base-triangle.png` - Green triangle on white background

Generated images use these shapes with randomized colors and backgrounds.

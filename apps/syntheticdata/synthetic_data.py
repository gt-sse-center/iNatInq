#!/usr/bin/env python3
"""Synthetic data generation and upload for pipeline testing.

This module provides tools to:
1. Generate test text documents from public domain books (e.g., Moby Dick)
2. Generate test images with semantic content (shapes + colors)
3. Upload generated data to MinIO/S3

Usage:
    # Text operations
    uv run python synthetic_data.py generate-text --count 1000
    uv run python synthetic_data.py upload-text
    uv run python synthetic_data.py setup-text --count 1000

    # Image operations
    uv run python synthetic_data.py generate-images --count 100
    uv run python synthetic_data.py upload-images
    uv run python synthetic_data.py setup-images --count 100

    # All data
    uv run python synthetic_data.py setup-all --count 1000 --image-count 100
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from typing import Any

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow not installed. Install with: uv add pillow", file=sys.stderr)
    sys.exit(1)

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 not installed. Install with: uv add boto3", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:
    print("tqdm not installed. Install with: uv add tqdm", file=sys.stderr)
    sys.exit(1)

# Import retry logic from foundation
try:
    from foundation.retry import RetryWithBackoff
except ImportError:
    # Fallback if running outside package context
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    from foundation.retry import RetryWithBackoff


# =============================================================================
# Constants
# =============================================================================

# Color definitions with names for semantic filenames
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 53, 69),
    "green": (40, 167, 69),
    "blue": (0, 123, 255),
    "yellow": (255, 193, 7),
    "orange": (253, 126, 20),
    "purple": (111, 66, 193),
    "pink": (232, 62, 140),
    "teal": (32, 201, 151),
}

# Shape names for semantic filenames
SHAPES = ["circle", "square", "triangle"]

# Background styles
BACKGROUNDS = ["solid", "gradient"]


# =============================================================================
# Text Generator
# =============================================================================


@attrs.define
class TextGenerator:
    """Generates test documents from a source text file.

    Splits source text into chunks and saves them as individual files.
    Useful for creating test data for ML pipelines.

    Note:
        This implementation is tailored for Project Gutenberg texts (e.g., Moby Dick)
        and includes logic to strip Gutenberg-specific headers/footers. This is
        intended for demo purposes; production use will likely require different
        data sources (e.g., iNaturalist images).

    Attributes:
        source_file: Path to source text file (e.g., Moby Dick).
        output_dir: Directory to write generated documents.
        chunk_size: Target size for each chunk in characters.
    """

    source_file: Path = attrs.field(converter=Path)
    output_dir: Path = attrs.field(converter=Path)
    chunk_size: int = attrs.field(default=500)

    def _clean_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers and footers.

        Args:
            text: Raw text from Project Gutenberg.

        Returns:
            Cleaned text with headers/footers removed.
        """
        # Remove Project Gutenberg header (everything before "CHAPTER 1")
        match = re.search(r"(?i)(?:^|\n)(?:chapter\s+1|chapter\s+i)\b", text)
        if match:
            text = text[match.start() :]

        # Remove Project Gutenberg footer
        match = re.search(r"(?i)\n\s*end\s+of\s+(?:the\s+)?project\s+gutenberg", text)
        if match:
            text = text[: match.start()]

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks of approximately chunk_size characters.

        Tries to break at sentence boundaries when possible.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.

        Note:
            TODO: Consider using nltk.sent_tokenize() for more robust sentence
            boundary detection. Skipped for now since we're moving to image data.
        """
        chunks = []
        current_pos = 0
        text_len = len(text)

        while current_pos < text_len:
            end_pos = min(current_pos + self.chunk_size, text_len)

            # Try to break at a sentence boundary
            if end_pos < text_len:
                search_start = max(current_pos, end_pos - self.chunk_size // 5)
                match = re.search(r"[.!?]\s+[A-Z]|[.!?]\n", text[search_start:end_pos])
                if match:
                    end_pos = search_start + match.end()

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos

        return chunks

    def generate(self, count: int) -> int:
        """Generate test documents from source text.

        Args:
            count: Number of documents to generate.

        Returns:
            Number of documents generated.

        Raises:
            FileNotFoundError: If source file does not exist.
        """
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_file}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Reading source file: {self.source_file}")
        with open(self.source_file, encoding="utf-8") as f:
            raw_text = f.read()

        print("Cleaning text...")
        cleaned_text = self._clean_text(raw_text)

        print(f"Source text length: {len(cleaned_text):,} characters")
        print(f"Generating {count} documents with ~{self.chunk_size} characters each...")

        # Split into chunks
        chunks = self._split_into_chunks(cleaned_text)

        # Generate documents (cycle through chunks if needed)
        documents_generated = 0
        chunk_index = 0

        while documents_generated < count:
            chunk = chunks[chunk_index % len(chunks)]
            output_file = self.output_dir / f"moby-dick-{documents_generated:05d}.txt"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(chunk)

            documents_generated += 1
            chunk_index += 1

            if documents_generated % 100 == 0:
                print(f"  Generated {documents_generated}/{count} documents...")

        print(f"Generated {documents_generated} documents in {self.output_dir}")
        return documents_generated


# =============================================================================
# Image Generator
# =============================================================================


@attrs.define
class ImageGenerator:
    """Generates synthetic test images with semantic content.

    Creates images with:
    - Colored backgrounds (solid or gradient)
    - Simple shapes (circles, squares, triangles)
    - Deterministic output via random seed

    Filenames encode content: {color}-{shape}-{background}-{index}.png

    Attributes:
        output_dir: Directory to write generated images.
        size: Image dimensions (width=height).
        seed: Random seed for deterministic output.
    """

    output_dir: Path = attrs.field(converter=Path)
    size: int = attrs.field(default=512)
    seed: int = attrs.field(default=42)

    def _draw_shape(
        self,
        draw: ImageDraw.ImageDraw,
        shape: str,
        color: tuple[int, int, int],
        center_x: int,
        center_y: int,
        radius: int,
    ) -> None:
        """Draw a shape on the image.

        Args:
            draw: ImageDraw object to draw on.
            shape: Shape type ('circle', 'square', 'triangle').
            color: RGB color tuple.
            center_x: X coordinate of shape center.
            center_y: Y coordinate of shape center.
            radius: Size of the shape (radius for circle, half-width for others).
        """
        if shape == "circle":
            draw.ellipse(
                [
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ],
                fill=color,
            )
        elif shape == "square":
            draw.rectangle(
                [
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ],
                fill=color,
            )
        elif shape == "triangle":
            # Equilateral triangle pointing up
            points = [
                (center_x, center_y - radius),  # top
                (center_x - radius, center_y + int(radius * 0.866)),  # bottom-left
                (center_x + radius, center_y + int(radius * 0.866)),  # bottom-right
            ]
            draw.polygon(points, fill=color)

    def _create_background(
        self,
        img: Image.Image,
        bg_style: str,
        bg_color: tuple[int, int, int],
    ) -> None:
        """Create background for the image.

        Args:
            img: PIL Image to draw on.
            bg_style: Background style ('solid' or 'gradient').
            bg_color: Primary background color.
        """
        draw = ImageDraw.Draw(img)

        if bg_style == "solid":
            draw.rectangle([0, 0, self.size, self.size], fill=bg_color)
        else:  # gradient
            # Create vertical gradient from bg_color to lighter version
            lighter = tuple(min(255, c + 80) for c in bg_color)
            for y in range(self.size):
                ratio = y / self.size
                r = int(bg_color[0] * (1 - ratio) + lighter[0] * ratio)
                g = int(bg_color[1] * (1 - ratio) + lighter[1] * ratio)
                b = int(bg_color[2] * (1 - ratio) + lighter[2] * ratio)
                draw.line([(0, y), (self.size, y)], fill=(r, g, b))

    def _generate_single_image(
        self,
        index: int,
        rng: random.Random,
    ) -> tuple[Image.Image, str]:
        """Generate a single image with random semantic content.

        Args:
            index: Image index for filename.
            rng: Random number generator for determinism.

        Returns:
            Tuple of (PIL Image, filename).
        """
        # Pick semantic elements
        shape_name = rng.choice(SHAPES)
        shape_color_name = rng.choice(list(COLORS.keys()))
        bg_style = rng.choice(BACKGROUNDS)

        # Pick background color (different from shape color)
        bg_colors = [c for c in COLORS if c != shape_color_name]
        bg_color_name = rng.choice(bg_colors)

        # Create image
        img = Image.new("RGB", (self.size, self.size))

        # Draw background
        self._create_background(img, bg_style, COLORS[bg_color_name])

        # Draw shape at center with some random offset
        center_x = self.size // 2 + rng.randint(-50, 50)
        center_y = self.size // 2 + rng.randint(-50, 50)
        radius = self.size // 4 + rng.randint(-30, 30)

        draw = ImageDraw.Draw(img)
        self._draw_shape(
            draw,
            shape_name,
            COLORS[shape_color_name],
            center_x,
            center_y,
            radius,
        )

        # Create semantic filename: color-shape-background-index.png
        filename = f"{shape_color_name}-{shape_name}-{bg_style}-{index:05d}.png"

        return img, filename

    def generate(self, count: int) -> int:
        """Generate test images.

        Args:
            count: Number of images to generate.

        Returns:
            Number of images generated.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {count} images at {self.size}x{self.size}...")
        print(f"Using seed: {self.seed}")

        rng = random.Random(self.seed)  # noqa: S311 - not for crypto, just test data
        images_generated = 0

        for i in range(count):
            img, filename = self._generate_single_image(i, rng)
            output_path = self.output_dir / filename
            img.save(output_path, "PNG")
            images_generated += 1

            if images_generated % 100 == 0:
                print(f"  Generated {images_generated}/{count} images...")

        print(f"Generated {images_generated} images in {self.output_dir}")
        return images_generated


# =============================================================================
# Unified MinIO Uploader
# =============================================================================


@attrs.define
class MinIOUploader:
    """Uploads files to MinIO/S3-compatible storage.

    Uses concurrent uploads with ThreadPoolExecutor for better performance.
    Includes retry logic and progress tracking. Automatically detects
    content type based on file extension.

    Attributes:
        endpoint: MinIO endpoint URL.
        access_key: MinIO access key.
        secret_key: MinIO secret key.
        bucket: Target bucket name.
        max_workers: Maximum concurrent upload threads.
        max_retries: Maximum retry attempts per file.
    """

    endpoint: str = attrs.field(default="http://localhost:9000")
    access_key: str = attrs.field(default="minioadmin")
    secret_key: str = attrs.field(default="minioadmin")
    bucket: str = attrs.field(default="pipeline")
    max_workers: int = attrs.field(default=50)
    max_retries: int = attrs.field(default=3)

    _client: Any = attrs.field(init=False, default=None)
    _retry: RetryWithBackoff = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        """Initialize S3 client and retry utility."""
        s3_config = Config(max_pool_connections=self.max_workers * 2)
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="us-east-1",
            config=s3_config,
        )
        self._retry = RetryWithBackoff(
            max_attempts=self.max_retries,
            wait_min=0.1,
            wait_max=2.0,
            multiplier=1.0,
        )

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type based on file extension.

        Args:
            file_path: Path to the file.

        Returns:
            MIME type string.
        """
        suffix = file_path.suffix.lower()
        content_types = {
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".json": "application/json",
        }
        return content_types.get(suffix, "application/octet-stream")

    def _upload_single_file(self, file_path: Path, s3_key: str) -> tuple[str, bool]:
        """Upload a single file with retry logic.

        Args:
            file_path: Path to file to upload.
            s3_key: S3 key (path) for the object.

        Returns:
            Tuple of (filename, success).
        """
        filename = file_path.name
        content_type = self._get_content_type(file_path)

        def _upload() -> None:
            self._client.upload_file(
                str(file_path),
                self.bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )

        try:
            self._retry.call(_upload, retry_exceptions=(ClientError, OSError))
            return (filename, True)
        except (ClientError, OSError):
            return (filename, False)

    def upload_directory(
        self,
        input_dir: Path,
        prefix: str = "inputs/",
        pattern: str = "*.txt",
    ) -> tuple[int, int]:
        """Upload all matching files from a directory to MinIO.

        Args:
            input_dir: Directory containing files to upload.
            prefix: S3 prefix (directory) to upload to.
            pattern: Glob pattern for files to upload.

        Returns:
            Tuple of (successful_count, failed_count).

        Raises:
            FileNotFoundError: If input directory does not exist.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files = sorted(input_dir.glob(pattern))
        if not files:
            print(f"No {pattern} files found in {input_dir}")
            return (0, 0)

        total_files = len(files)
        print(f"Found {total_files} files to upload...")
        print(f"Uploading to s3://{self.bucket}/{prefix}")
        print(f"Using {self.max_workers} concurrent workers...")

        progress_bar = tqdm(
            total=total_files,
            desc="Uploading",
            unit="files",
            mininterval=1.0,
            maxinterval=10.0,
            file=sys.stdout,
            ncols=100,
        )

        successful = 0
        failed = 0
        start_time = time.time()

        def upload_with_progress(file_path: Path) -> tuple[str, bool]:
            s3_key = f"{prefix.rstrip('/')}/{file_path.name}"
            result = self._upload_single_file(file_path, s3_key)
            progress_bar.update(1)
            return result

        try:
            with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {executor.submit(upload_with_progress, fp): fp for fp in files}

                for future in futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        filename, success = future.result()
                        if success:
                            successful += 1
                        else:
                            failed += 1
                            print(f"  Failed: {filename}", file=sys.stderr)
                    except (ClientError, OSError, RuntimeError) as e:
                        failed += 1
                        print(f"  Exception: {file_path.name}: {e}", file=sys.stderr)
        finally:
            progress_bar.close()

        elapsed = time.time() - start_time
        rate = total_files / elapsed if elapsed > 0 else 0

        print(f"\nUpload complete: {successful} successful, {failed} failed ")
        print(f"   ({elapsed:.1f}s, {rate:.1f} files/s)")

        return (successful, failed)


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_generate_text(args: argparse.Namespace) -> int:
    """Handle 'generate-text' subcommand."""
    generator = TextGenerator(
        source_file=args.source,
        output_dir=args.output,
        chunk_size=args.chunk_size,
    )
    try:
        generator.generate(count=args.count)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_upload_text(args: argparse.Namespace) -> int:
    """Handle 'upload-text' subcommand."""
    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.input_dir,
            prefix=args.prefix,
            pattern="*.txt",
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_setup_text(args: argparse.Namespace) -> int:
    """Handle 'setup-text' subcommand (generate + upload text)."""
    # Generate
    generator = TextGenerator(
        source_file=args.source,
        output_dir=args.output,
        chunk_size=args.chunk_size,
    )
    try:
        generator.generate(count=args.count)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()  # Blank line between steps

    # Upload
    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.output,
            prefix=args.prefix,
            pattern="*.txt",
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_generate_images(args: argparse.Namespace) -> int:
    """Handle 'generate-images' subcommand."""
    generator = ImageGenerator(
        output_dir=args.output,
        size=args.size,
        seed=args.seed,
    )
    generator.generate(count=args.count)
    return 0


def cmd_upload_images(args: argparse.Namespace) -> int:
    """Handle 'upload-images' subcommand."""
    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.input_dir,
            prefix=args.prefix,
            pattern="*.png",
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_setup_images(args: argparse.Namespace) -> int:
    """Handle 'setup-images' subcommand (generate + upload images)."""
    # Generate
    generator = ImageGenerator(
        output_dir=args.output,
        size=args.size,
        seed=args.seed,
    )
    generator.generate(count=args.count)

    print()  # Blank line between steps

    # Upload
    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.output,
            prefix=args.prefix,
            pattern="*.png",
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_setup_all(args: argparse.Namespace) -> int:
    """Handle 'setup-all' subcommand (generate + upload both text and images)."""
    # Generate and upload text
    print("=" * 60)
    print("Generating and uploading text documents...")
    print("=" * 60)

    text_generator = TextGenerator(
        source_file=args.text_source,
        output_dir=args.text_output,
        chunk_size=args.chunk_size,
    )
    try:
        text_generator.generate(count=args.count)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()

    uploader = MinIOUploader(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.text_output,
            prefix=args.text_prefix,
            pattern="*.txt",
        )
        if failed > 0:
            return 1
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()
    print("=" * 60)
    print("Generating and uploading images...")
    print("=" * 60)

    # Generate and upload images
    image_generator = ImageGenerator(
        output_dir=args.image_output,
        size=args.size,
        seed=args.seed,
    )
    image_generator.generate(count=args.image_count)

    print()

    try:
        _successful, failed = uploader.upload_directory(
            input_dir=args.image_output,
            prefix=args.image_prefix,
            pattern="*.png",
        )
        return 1 if failed > 0 else 0
    except (FileNotFoundError, ClientError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# =============================================================================
# CLI Argument Parsers
# =============================================================================


def add_minio_args(parser: argparse.ArgumentParser) -> None:
    """Add common MinIO arguments to a parser."""
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:9000",
        help="MinIO endpoint (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        default="minioadmin",
        help="MinIO access key (default: minioadmin)",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        default="minioadmin",
        help="MinIO secret key (default: minioadmin)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="pipeline",
        help="MinIO bucket name (default: pipeline)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Concurrent upload threads (default: 50)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per file (default: 3)",
    )


def main() -> None:
    """Main CLI entry point."""
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Synthetic data generation and upload for pipeline testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate-text subcommand ---
    gen_text = subparsers.add_parser("generate-text", help="Generate test text documents")
    gen_text.add_argument(
        "--source",
        type=Path,
        default=base_dir / "seed" / "txt" / "moby-dick.txt",
        help="Source text file (default: seed/txt/moby-dick.txt)",
    )
    gen_text.add_argument(
        "--output",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Output directory (default: data/txts/)",
    )
    gen_text.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of documents to generate (default: 1000)",
    )
    gen_text.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    gen_text.set_defaults(func=cmd_generate_text)

    # --- upload-text subcommand ---
    up_text = subparsers.add_parser("upload-text", help="Upload text documents to MinIO")
    up_text.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Directory with files to upload (default: data/txts/)",
    )
    up_text.add_argument(
        "--prefix",
        type=str,
        default="inputs/",
        help="S3 prefix/directory (default: inputs/)",
    )
    add_minio_args(up_text)
    up_text.set_defaults(func=cmd_upload_text)

    # --- setup-text subcommand ---
    setup_text = subparsers.add_parser("setup-text", help="Generate and upload text (all-in-one)")
    setup_text.add_argument(
        "--source",
        type=Path,
        default=base_dir / "seed" / "txt" / "moby-dick.txt",
        help="Source text file (default: seed/txt/moby-dick.txt)",
    )
    setup_text.add_argument(
        "--output",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Output directory (default: data/txts/)",
    )
    setup_text.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of documents to generate (default: 1000)",
    )
    setup_text.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    setup_text.add_argument(
        "--prefix",
        type=str,
        default="inputs/",
        help="S3 prefix/directory (default: inputs/)",
    )
    add_minio_args(setup_text)
    setup_text.set_defaults(func=cmd_setup_text)

    # --- generate-images subcommand ---
    gen_images = subparsers.add_parser("generate-images", help="Generate test images")
    gen_images.add_argument(
        "--output",
        type=Path,
        default=base_dir / "data" / "imgs",
        help="Output directory (default: data/imgs/)",
    )
    gen_images.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to generate (default: 100)",
    )
    gen_images.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image size in pixels (default: 512)",
    )
    gen_images.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42)",
    )
    gen_images.set_defaults(func=cmd_generate_images)

    # --- upload-images subcommand ---
    up_images = subparsers.add_parser("upload-images", help="Upload images to MinIO")
    up_images.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "data" / "imgs",
        help="Directory with files to upload (default: data/imgs/)",
    )
    up_images.add_argument(
        "--prefix",
        type=str,
        default="images/",
        help="S3 prefix/directory (default: images/)",
    )
    add_minio_args(up_images)
    up_images.set_defaults(func=cmd_upload_images)

    # --- setup-images subcommand ---
    setup_images = subparsers.add_parser("setup-images", help="Generate and upload images (all-in-one)")
    setup_images.add_argument(
        "--output",
        type=Path,
        default=base_dir / "data" / "imgs",
        help="Output directory (default: data/imgs/)",
    )
    setup_images.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to generate (default: 100)",
    )
    setup_images.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image size in pixels (default: 512)",
    )
    setup_images.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42)",
    )
    setup_images.add_argument(
        "--prefix",
        type=str,
        default="images/",
        help="S3 prefix/directory (default: images/)",
    )
    add_minio_args(setup_images)
    setup_images.set_defaults(func=cmd_setup_images)

    # --- setup-all subcommand ---
    setup_all = subparsers.add_parser("setup-all", help="Generate and upload both text and images")
    # Text options
    setup_all.add_argument(
        "--text-source",
        type=Path,
        default=base_dir / "seed" / "txt" / "moby-dick.txt",
        help="Source text file (default: seed/txt/moby-dick.txt)",
    )
    setup_all.add_argument(
        "--text-output",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Text output directory (default: data/txts/)",
    )
    setup_all.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of text documents to generate (default: 1000)",
    )
    setup_all.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    setup_all.add_argument(
        "--text-prefix",
        type=str,
        default="inputs/",
        help="S3 prefix for text (default: inputs/)",
    )
    # Image options
    setup_all.add_argument(
        "--image-output",
        type=Path,
        default=base_dir / "data" / "imgs",
        help="Image output directory (default: data/imgs/)",
    )
    setup_all.add_argument(
        "--image-count",
        type=int,
        default=100,
        help="Number of images to generate (default: 100)",
    )
    setup_all.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image size in pixels (default: 512)",
    )
    setup_all.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42)",
    )
    setup_all.add_argument(
        "--image-prefix",
        type=str,
        default="images/",
        help="S3 prefix for images (default: images/)",
    )
    add_minio_args(setup_all)
    setup_all.set_defaults(func=cmd_setup_all)

    # --- Backwards-compatible aliases ---
    # generate (alias for generate-text)
    gen_alias = subparsers.add_parser("generate", help="[Deprecated] Alias for generate-text")
    gen_alias.add_argument(
        "--source",
        type=Path,
        default=base_dir / "seed" / "txt" / "moby-dick.txt",
        help="Source text file",
    )
    gen_alias.add_argument(
        "--output",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Output directory",
    )
    gen_alias.add_argument("--count", type=int, default=1000)
    gen_alias.add_argument("--chunk-size", type=int, default=500)
    gen_alias.set_defaults(func=cmd_generate_text)

    # upload (alias for upload-text)
    up_alias = subparsers.add_parser("upload", help="[Deprecated] Alias for upload-text")
    up_alias.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Directory with files to upload",
    )
    up_alias.add_argument("--prefix", type=str, default="inputs/")
    add_minio_args(up_alias)
    up_alias.set_defaults(func=cmd_upload_text)

    # setup (alias for setup-text)
    setup_alias = subparsers.add_parser("setup", help="[Deprecated] Alias for setup-text")
    setup_alias.add_argument(
        "--source",
        type=Path,
        default=base_dir / "seed" / "txt" / "moby-dick.txt",
        help="Source text file",
    )
    setup_alias.add_argument(
        "--output",
        type=Path,
        default=base_dir / "data" / "txts",
        help="Output directory",
    )
    setup_alias.add_argument("--count", type=int, default=1000)
    setup_alias.add_argument("--chunk-size", type=int, default=500)
    setup_alias.add_argument("--prefix", type=str, default="inputs/")
    add_minio_args(setup_alias)
    setup_alias.set_defaults(func=cmd_setup_text)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

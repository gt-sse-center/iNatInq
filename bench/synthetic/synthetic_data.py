#!/usr/bin/env python3
"""Synthetic data generation and upload for pipeline testing.

This module provides tools to:
1. Generate test images with semantic content (shapes + colors)
2. Upload generated data to MinIO/S3

Usage via CLI:
    uv run inq synthetic generate --count 100
    uv run inq synthetic upload
    uv run inq synthetic setup --count 100
"""

from __future__ import annotations

import random
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

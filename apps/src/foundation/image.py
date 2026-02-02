"""Image preprocessing utilities for embedding pipelines.

This module provides utilities for preprocessing images before embedding:
- Resizing to model-expected dimensions while preserving aspect ratio
- Converting to RGB format (handles RGBA, grayscale, etc.)
- Image validation and format detection

These utilities are used by the ingestion pipeline to ensure images are in
the correct format for embedding models like CLIP.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

logger = logging.getLogger("pipeline.ingestion.image")


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    r"""Validate image bytes and detect format.

    Checks if the provided bytes represent a valid image that can be
    processed by PIL. Returns validation status and error message.

    Args:
        image_bytes: Raw image bytes to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
        If invalid, error_message contains a description of the issue.

    Example:
        >>> valid, error = validate_image(b"\xff\xd8\xff...")
        >>> if valid:
        ...     print("Image is valid")
        >>> else:
        ...     print(f"Invalid: {error}")
    """
    if not image_bytes:
        return (False, "Image bytes cannot be empty")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Verify the image can be loaded and has valid format
            img.verify()
        return (True, "")
    except Exception as e:
        error_msg = f"Invalid image format: {e!s}"
        logger.debug("Image validation failed", extra={"error": str(e), "error_type": type(e).__name__})
        return (False, error_msg)


def resize_for_embedding(image_bytes: bytes, max_size: int = 224) -> bytes:
    """Resize image to model-expected size while preserving aspect ratio.

    This function:
    - Converts image to RGB (handles RGBA, grayscale, etc.)
    - Resizes to fit within max_size x max_size while preserving aspect ratio
    - Pads with black pixels if needed to create square output
    - Returns processed image as JPEG bytes

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, WebP, GIF, etc.).
        max_size: Maximum dimension for the output image. Default: 224 (common
            for CLIP models like ViT-B/32).

    Returns:
        Processed image bytes in JPEG format, resized and converted to RGB.

    Raises:
        ValueError: If image_bytes is empty or invalid.

    Example:
        >>> with open("photo.jpg", "rb") as f:
        ...     original = f.read()
        >>> processed = resize_for_embedding(original, max_size=224)
        >>> # processed is now 224x224 RGB JPEG ready for embedding
    """
    if not image_bytes:
        msg = "Image bytes cannot be empty"
        raise ValueError(msg)

    try:
        # Open and convert to RGB
        with Image.open(io.BytesIO(image_bytes)) as original_img:
            # Convert to RGB (handles RGBA, grayscale, palette, etc.)
            rgb_img = original_img.convert("RGB") if original_img.mode != "RGB" else original_img.copy()

            # Calculate resize dimensions preserving aspect ratio
            width, height = rgb_img.size
            if width > max_size or height > max_size:
                # Resize to fit within max_size x max_size
                rgb_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                width, height = rgb_img.size

            # Create square image with black padding if needed
            if width != max_size or height != max_size:
                # Create new square image with black background
                square_img = Image.new("RGB", (max_size, max_size), (0, 0, 0))
                # Calculate centering position
                x_offset = (max_size - width) // 2
                y_offset = (max_size - height) // 2
                # Paste resized image onto center of square
                square_img.paste(rgb_img, (x_offset, y_offset))
                final_img = square_img
            else:
                final_img = rgb_img

            # Convert to bytes (JPEG format)
            output = io.BytesIO()
            final_img.save(output, format="JPEG", quality=95)
            return output.getvalue()

    except Exception as e:
        error_msg = f"Failed to resize image: {e!s}"
        logger.exception("Image resize failed", extra={"error": str(e), "error_type": type(e).__name__})
        raise ValueError(error_msg) from e

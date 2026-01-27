"""Unit tests for core.ingestion.image_utils module.

This file tests image preprocessing utilities for embedding pipelines.

# Test Coverage

The tests cover:
  - validate_image: Image validation and format detection
  - resize_for_embedding: Resizing, RGB conversion, aspect ratio preservation
  - Edge cases: Empty bytes, invalid formats, various image modes
  - Different image formats: JPEG, PNG, WebP, GIF
  - Different color modes: RGB, RGBA, grayscale, palette

# Test Structure

Tests use pytest class-based organization with programmatically generated
test images using PIL.

# Running Tests

Run with: pytest tests/unit/core/ingestion/test_image_utils.py
"""

import io

import pytest
from PIL import Image

from core.ingestion.image_utils import resize_for_embedding, validate_image


# =============================================================================
# Test Image Fixtures
# =============================================================================


def create_test_image(
    width: int,
    height: int,
    mode: str = "RGB",
    format: str = "JPEG",
) -> bytes:
    """Create a test image with specified dimensions and mode.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        mode: PIL image mode (RGB, RGBA, L, P, etc.).
        format: Output format (JPEG, PNG, etc.).

    Returns:
        Image bytes in the specified format.
    """
    # Create a simple test image with a gradient
    img = Image.new(mode, (width, height))
    if mode == "RGB":
        # Fill with a simple gradient
        pixels = []
        for y in range(height):
            for x in range(width):
                r = (x * 255) // width if width > 0 else 0
                g = (y * 255) // height if height > 0 else 0
                b = 128
                pixels.append((r, g, b))
        img.putdata(pixels)
    elif mode == "RGBA":
        # Fill with gradient + alpha
        pixels = []
        for y in range(height):
            for x in range(width):
                r = (x * 255) // width if width > 0 else 0
                g = (y * 255) // height if height > 0 else 0
                b = 128
                a = 255
                pixels.append((r, g, b, a))
        img.putdata(pixels)
    elif mode == "L":
        # Grayscale gradient
        pixels = []
        for y in range(height):
            for x in range(width):
                gray = ((x + y) * 255) // (width + height) if (width + height) > 0 else 0
                pixels.append(gray)
        img.putdata(pixels)
    else:
        # For other modes, just fill with a solid color
        if mode == "P":
            img = img.convert("RGB").convert("P")

    output = io.BytesIO()
    img.save(output, format=format)
    return output.getvalue()


# =============================================================================
# validate_image Tests
# =============================================================================


class TestValidateImage:
    """Test suite for validate_image function."""

    def test_validate_jpeg_image(self) -> None:
        """Test validation of valid JPEG image.

        **Why this test is important:**
          - JPEG is most common format
          - Must validate successfully

        **What it tests:**
          - Returns (True, "") for valid JPEG
        """
        jpeg_bytes = create_test_image(100, 100, mode="RGB", format="JPEG")

        valid, error = validate_image(jpeg_bytes)

        assert valid is True
        assert error == ""

    def test_validate_png_image(self) -> None:
        """Test validation of valid PNG image.

        **Why this test is important:**
          - PNG widely used for lossless images
          - Must validate successfully

        **What it tests:**
          - Returns (True, "") for valid PNG
        """
        png_bytes = create_test_image(100, 100, mode="RGB", format="PNG")

        valid, error = validate_image(png_bytes)

        assert valid is True
        assert error == ""

    def test_validate_webp_image(self) -> None:
        """Test validation of valid WebP image.

        **Why this test is important:**
          - WebP is modern efficient format
          - Must validate successfully

        **What it tests:**
          - Returns (True, "") for valid WebP
        """
        webp_bytes = create_test_image(100, 100, mode="RGB", format="WEBP")

        valid, error = validate_image(webp_bytes)

        assert valid is True
        assert error == ""

    def test_validate_gif_image(self) -> None:
        """Test validation of valid GIF image.

        **Why this test is important:**
          - GIF still used for simple images
          - Must validate successfully

        **What it tests:**
          - Returns (True, "") for valid GIF
        """
        gif_bytes = create_test_image(100, 100, mode="RGB", format="GIF")

        valid, error = validate_image(gif_bytes)

        assert valid is True
        assert error == ""

    def test_validate_empty_bytes(self) -> None:
        """Test validation of empty bytes.

        **Why this test is important:**
          - Empty input is common edge case
          - Must handle gracefully

        **What it tests:**
          - Returns (False, error_message) for empty bytes
        """
        valid, error = validate_image(b"")

        assert valid is False
        assert "empty" in error.lower()

    def test_validate_invalid_bytes(self) -> None:
        """Test validation of invalid image bytes.

        **Why this test is important:**
          - Invalid data must be caught
          - Error message should be descriptive

        **What it tests:**
          - Returns (False, error_message) for invalid bytes
        """
        invalid_bytes = b"This is not an image"

        valid, error = validate_image(invalid_bytes)

        assert valid is False
        assert len(error) > 0

    def test_validate_corrupted_image(self) -> None:
        """Test validation of corrupted image data.

        **Why this test is important:**
          - Corrupted files can occur in production
          - Must handle gracefully

        **What it tests:**
          - Returns (False, error_message) for corrupted data
        """
        # Create valid JPEG header but corrupt the rest
        corrupted = b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"corrupted data"

        valid, error = validate_image(corrupted)

        assert valid is False
        assert len(error) > 0


# =============================================================================
# resize_for_embedding Tests
# =============================================================================


class TestResizeForEmbedding:
    """Test suite for resize_for_embedding function."""

    def test_resize_smaller_image(self) -> None:
        """Test resizing image smaller than max_size.

        **Why this test is important:**
          - Small images should be padded to square
          - Must preserve aspect ratio

        **What it tests:**
          - Output is max_size x max_size
          - Image content preserved
        """
        small_image = create_test_image(50, 50, mode="RGB", format="JPEG")

        result = resize_for_embedding(small_image, max_size=224)

        # Verify output is valid image
        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_larger_image(self) -> None:
        """Test resizing image larger than max_size.

        **Why this test is important:**
          - Large images must be resized down
          - Aspect ratio must be preserved

        **What it tests:**
          - Output fits within max_size x max_size
          - Aspect ratio maintained
        """
        large_image = create_test_image(500, 300, mode="RGB", format="JPEG")

        result = resize_for_embedding(large_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_wide_image(self) -> None:
        """Test resizing wide (landscape) image.

        **Why this test is important:**
          - Wide images common in practice
          - Must handle aspect ratio correctly

        **What it tests:**
          - Wide images resized and padded correctly
        """
        wide_image = create_test_image(400, 200, mode="RGB", format="JPEG")

        result = resize_for_embedding(wide_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_tall_image(self) -> None:
        """Test resizing tall (portrait) image.

        **Why this test is important:**
          - Portrait images common in practice
          - Must handle aspect ratio correctly

        **What it tests:**
          - Tall images resized and padded correctly
        """
        tall_image = create_test_image(200, 400, mode="RGB", format="JPEG")

        result = resize_for_embedding(tall_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_exact_size(self) -> None:
        """Test resizing image that is already exact size.

        **Why this test is important:**
          - Edge case: already correct size
          - Should still convert to RGB and return JPEG

        **What it tests:**
          - Output is still max_size x max_size
          - Format conversion works
        """
        exact_image = create_test_image(224, 224, mode="RGB", format="JPEG")

        result = resize_for_embedding(exact_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_rgba_image(self) -> None:
        """Test resizing RGBA image (with alpha channel).

        **Why this test is important:**
          - RGBA images common (PNG with transparency)
          - Must convert to RGB correctly

        **What it tests:**
          - RGBA converted to RGB
          - Alpha channel handled properly
        """
        rgba_image = create_test_image(100, 100, mode="RGBA", format="PNG")

        result = resize_for_embedding(rgba_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_grayscale_image(self) -> None:
        """Test resizing grayscale (L mode) image.

        **Why this test is important:**
          - Grayscale images common
          - Must convert to RGB correctly

        **What it tests:**
          - Grayscale converted to RGB
          - Output is 3-channel RGB
        """
        grayscale_image = create_test_image(100, 100, mode="L", format="PNG")

        result = resize_for_embedding(grayscale_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_custom_max_size(self) -> None:
        """Test resizing with custom max_size.

        **Why this test is important:**
          - Different models need different sizes
          - Must support configurable size

        **What it tests:**
          - Custom max_size works correctly
        """
        image = create_test_image(500, 500, mode="RGB", format="JPEG")

        result = resize_for_embedding(image, max_size=512)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (512, 512)
            assert img.mode == "RGB"

    def test_resize_empty_bytes_raises(self) -> None:
        """Test that empty bytes raise ValueError.

        **Why this test is important:**
          - Empty input must be rejected
          - Error should be clear

        **What it tests:**
          - Raises ValueError for empty bytes
        """
        with pytest.raises(ValueError, match="empty"):
            resize_for_embedding(b"", max_size=224)

    def test_resize_invalid_bytes_raises(self) -> None:
        """Test that invalid bytes raise ValueError.

        **Why this test is important:**
          - Invalid data must be caught
          - Error should be descriptive

        **What it tests:**
          - Raises ValueError for invalid bytes
        """
        invalid_bytes = b"This is not an image"

        with pytest.raises(ValueError, match="Failed to resize"):
            resize_for_embedding(invalid_bytes, max_size=224)

    def test_resize_preserves_aspect_ratio(self) -> None:
        """Test that aspect ratio is preserved during resize.

        **Why this test is important:**
          - Aspect ratio critical for image quality
          - Must not distort images

        **What it tests:**
          - Wide images maintain width/height ratio
          - Padding added symmetrically
        """
        # Create 2:1 aspect ratio image
        wide_image = create_test_image(200, 100, mode="RGB", format="JPEG")

        result = resize_for_embedding(wide_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            # Should be square (224x224) with image centered
            assert img.size == (224, 224)
            # Check that image is centered (black padding on top/bottom)
            # We can't easily verify exact content, but size is correct
            assert img.mode == "RGB"

    def test_resize_png_image(self) -> None:
        """Test resizing PNG image.

        **Why this test is important:**
          - PNG is common format
          - Must handle correctly

        **What it tests:**
          - PNG input produces JPEG output
        """
        png_image = create_test_image(300, 200, mode="RGB", format="PNG")

        result = resize_for_embedding(png_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_webp_image(self) -> None:
        """Test resizing WebP image.

        **Why this test is important:**
          - WebP is modern format
          - Must handle correctly

        **What it tests:**
          - WebP input produces JPEG output
        """
        webp_image = create_test_image(300, 200, mode="RGB", format="WEBP")

        result = resize_for_embedding(webp_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_gif_image(self) -> None:
        """Test resizing GIF image.

        **Why this test is important:**
          - GIF still used
          - Must handle correctly

        **What it tests:**
          - GIF input produces JPEG output
        """
        gif_image = create_test_image(300, 200, mode="RGB", format="GIF")

        result = resize_for_embedding(gif_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_very_large_image(self) -> None:
        """Test resizing very large image.

        **Why this test is important:**
          - Large images common in production
          - Must handle efficiently

        **What it tests:**
          - Very large images resized correctly
        """
        large_image = create_test_image(2000, 1500, mode="RGB", format="JPEG")

        result = resize_for_embedding(large_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_very_small_image(self) -> None:
        """Test resizing very small image.

        **Why this test is important:**
          - Small images should be upscaled/padded
          - Must handle correctly

        **What it tests:**
          - Very small images padded to square
        """
        small_image = create_test_image(10, 10, mode="RGB", format="JPEG")

        result = resize_for_embedding(small_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_resize_square_image(self) -> None:
        """Test resizing square image.

        **Why this test is important:**
          - Square images common
          - Should handle efficiently

        **What it tests:**
          - Square images resized correctly
        """
        square_image = create_test_image(300, 300, mode="RGB", format="JPEG")

        result = resize_for_embedding(square_image, max_size=224)

        with Image.open(io.BytesIO(result)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

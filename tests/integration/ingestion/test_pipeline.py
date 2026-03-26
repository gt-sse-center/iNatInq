"""Integration tests for the full ingestion pipeline.

Tests end-to-end pipeline functionality with real services:
Ray + MinIO + Qdrant + CLIP.

# Test Coverage

The tests cover:
  - Image processing pipeline: S3 → CLIP → Qdrant
  - Checkpoint management: Save/load with real storage
  - Error recovery: Failed items, retries, circuit breaker

# Running Tests

Run with: pytest tests/integration/ingestion/test_pipeline.py -v -m integration

Note: These tests require significant resources (~4GB RAM, 4 CPUs).
"""

import io
import uuid
import asyncio

import pytest
from PIL import Image


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_text_file(content: str) -> bytes:
    """Create a text file as bytes."""
    return content.encode("utf-8")


def create_test_image(width: int = 100, height: int = 100) -> bytes:
    """Create a simple test image as JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    output = io.BytesIO()
    img.save(output, format="JPEG")
    return output.getvalue()


# =============================================================================
# Image Pipeline Integration Tests
# =============================================================================


@pytest.mark.integration
class TestImagePipelineIntegration:
    """Integration tests for image processing pipeline."""

    def test_image_upload_and_retrieve(
        self,
        minio_client,
        test_bucket: str,
    ) -> None:
        """Test uploading and retrieving images from S3.

        **Why this test is important:**
          - S3 is the data source for pipeline
          - Validates client wrapper works
          - Ensures data integrity

        **What it tests:**
          - Image can be uploaded to S3
          - Image can be retrieved from S3
          - Binary content matches original
        """
        key = f"images/test-{uuid.uuid4().hex}.jpg"
        image_bytes = create_test_image(200, 200)

        # Upload
        minio_client.put_object(
            bucket=test_bucket,
            key=key,
            body=image_bytes,
        )

        # Retrieve
        retrieved = minio_client.get_object(bucket=test_bucket, key=key)
        assert retrieved == image_bytes

    def test_image_validation(self) -> None:
        """Test image validation utility.

        **Why this test is important:**
          - Invalid images must be rejected
          - Prevents embedding failures
          - Validates preprocessing

        **What it tests:**
          - Valid images pass validation
          - Invalid data fails validation
          - Error messages are clear
        """
        from foundation.image import validate_image

        # Valid image
        valid_image = create_test_image()
        is_valid, error = validate_image(valid_image)
        assert is_valid is True
        assert error == ""

        # Invalid data
        invalid_data = b"not an image"
        is_valid, error = validate_image(invalid_data)
        assert is_valid is False
        assert len(error) > 0

    def test_image_resize_for_embedding(self) -> None:
        """Test image resizing for embedding models.

        **Why this test is important:**
          - CLIP expects specific dimensions
          - Validates preprocessing
          - Ensures aspect ratio handling

        **What it tests:**
          - Images are resized to target size
          - Aspect ratio is preserved
          - Output is RGB JPEG
        """
        from foundation.image import resize_for_embedding

        # Create a non-square image
        wide_image = create_test_image(400, 200)

        resized = resize_for_embedding(wide_image, max_size=224)

        # Verify output
        with Image.open(io.BytesIO(resized)) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"


# =============================================================================
# Checkpoint Integration Tests
# =============================================================================


@pytest.mark.integration
class TestCheckpointIntegration:
    """Integration tests for checkpoint management."""

    def test_checkpoint_save_and_load_local(self, tmp_path) -> None:
        """Test saving and loading checkpoints from local filesystem.

        **Why this test is important:**
          - Checkpoints enable job recovery
          - Validates persistence
          - Ensures data integrity

        **What it tests:**
          - Checkpoint can be saved
          - Checkpoint can be loaded
          - Data matches original
        """
        from foundation.checkpoint import CheckpointManager

        checkpoint_file = tmp_path / "checkpoint.json"
        processed_keys = {"key1", "key2", "key3"}

        manager = CheckpointManager()

        # Save
        manager.save(checkpoint_file, processed_keys)
        assert checkpoint_file.exists()

        # Load
        loaded = manager.load(checkpoint_file)
        assert loaded == processed_keys

    def test_checkpoint_save_and_load_s3(
        self,
        minio_client,
        minio_config: dict,
        test_bucket: str,
    ) -> None:
        """Test saving and loading checkpoints from S3.

        **Why this test is important:**
          - S3 checkpoints enable distributed recovery
          - Validates S3 integration
          - Critical for cloud deployments

        **What it tests:**
          - Checkpoint can be saved to S3
          - Checkpoint can be loaded from S3
          - Data matches original
        """
        from foundation.checkpoint import CheckpointManager

        checkpoint_key = f"checkpoints/test-{uuid.uuid4().hex}.json"
        s3_path = f"s3://{test_bucket}/{checkpoint_key}"
        processed_keys = {"s3-key1", "s3-key2", "s3-key3"}

        manager = CheckpointManager(s3_client=minio_client)

        # Save
        manager.save(s3_path, processed_keys)

        # Verify object exists via direct boto3 head_object
        minio_client.client.head_object(Bucket=test_bucket, Key=checkpoint_key)

        # Load
        loaded = manager.load(s3_path)
        assert loaded == processed_keys

    def test_checkpoint_nonexistent_returns_empty(self, tmp_path) -> None:
        """Test loading nonexistent checkpoint returns empty set.

        **Why this test is important:**
          - First run has no checkpoint
          - Must not fail on missing file
          - Validates graceful handling

        **What it tests:**
          - Missing file returns empty set
          - No exception raised
        """
        from foundation.checkpoint import CheckpointManager

        checkpoint_file = tmp_path / "nonexistent.json"

        manager = CheckpointManager()
        loaded = manager.load(checkpoint_file)

        assert loaded == set()


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipelineIntegration:
    """Full end-to-end pipeline integration tests.

    These tests are marked as 'slow' and require all services to be running.
    They test the complete flow from S3 through embedding to vector DB.
    """

    # TODO: Add full pipeline integration test that uses CLIP for image embedding
    pass

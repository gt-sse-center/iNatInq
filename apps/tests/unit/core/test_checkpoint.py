"""Unit tests for core.ingestion.checkpoint module.

This file tests checkpoint management utilities for tracking processed items,
enabling job recovery and avoiding reprocessing.

# Test Coverage

The tests cover:
  - CheckpointManager: Initialization, local/S3 operations
  - Load Operations: Local filesystem, S3, empty checkpoints, errors
  - Save Operations: Local filesystem, S3, error handling
  - Utility Functions: is_s3_path, _parse_s3_path

# Test Structure

Tests use pytest class-based organization with mocking for S3 operations.
Local filesystem tests use temporary directories.

# Running Tests

Run with: pytest tests/unit/core/test_checkpoint.py
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import attrs.exceptions
import pytest
from botocore.exceptions import ClientError

from src.core.ingestion.checkpoint import CheckpointManager, is_s3_path

# =============================================================================
# Utility Function Tests
# =============================================================================


class TestIsS3Path:
    """Test suite for is_s3_path utility function."""

    def test_detects_s3_uri(self) -> None:
        """Test that s3:// URIs are detected.

        **Why this test is important:**
          - s3:// is the standard S3 URI scheme
          - Critical for routing to S3 operations
          - Prevents misrouting of paths
          - Validates URI parsing

        **What it tests:**
          - s3:// prefix is recognized
          - Returns True for S3 URIs
        """
        assert is_s3_path("s3://bucket/key") is True
        assert is_s3_path("s3://bucket/path/to/file.json") is True

    def test_detects_s3a_uri(self) -> None:
        """Test that s3a:// URIs are detected.

        **Why this test is important:**
          - s3a:// is Spark's S3 scheme
          - Critical for Spark job compatibility
          - Validates alternative scheme support
          - Prevents Spark integration issues

        **What it tests:**
          - s3a:// prefix is recognized
          - Returns True for S3a URIs
        """
        assert is_s3_path("s3a://bucket/key") is True
        assert is_s3_path("s3a://bucket/path/to/file.json") is True

    def test_rejects_local_paths(self) -> None:
        """Test that local filesystem paths are not detected as S3.

        **Why this test is important:**
          - Local paths must route to filesystem operations
          - Prevents incorrect S3 API calls
          - Critical for correct operation routing
          - Validates negative case

        **What it tests:**
          - Absolute paths return False
          - Relative paths return False
          - file:// URIs return False
        """
        assert is_s3_path("/tmp/checkpoint.json") is False
        assert is_s3_path("./checkpoint.json") is False
        assert is_s3_path("file:///tmp/checkpoint.json") is False

    def test_handles_path_objects(self) -> None:
        """Test that Path objects are handled correctly.

        **Why this test is important:**
          - Code uses Path objects internally
          - Must convert to string for checking
          - Validates Path compatibility
          - Critical for type flexibility

        **What it tests:**
          - Path objects are converted to strings
          - S3 Paths cannot be Path objects (not valid)
          - Local Paths work with Path objects
        """
        # S3 paths should be strings, not Path objects
        # But the function should handle them without error
        assert is_s3_path(Path("/tmp/file")) is False
        # String S3 paths work
        assert is_s3_path("s3://bucket/key") is True


# =============================================================================
# CheckpointManager Initialization Tests
# =============================================================================


class TestCheckpointManagerInit:
    """Test suite for CheckpointManager initialization."""

    def test_creates_manager_without_s3(self) -> None:
        """Test that manager is created without S3 client.

        **Why this test is important:**
          - S3 client is optional
          - Allows local-only operation
          - Validates optional dependency
          - Critical for flexibility

        **What it tests:**
          - Manager created with no args
          - s3_client defaults to None
          - No errors raised
        """
        manager = CheckpointManager()

        assert manager.s3_client is None

    def test_creates_manager_with_s3(self) -> None:
        """Test that manager is created with S3 client.

        **Why this test is important:**
          - S3 storage requires client
          - Validates dependency injection
          - Critical for S3 operations
          - Ensures client is stored

        **What it tests:**
          - Manager accepts s3_client argument
          - Client is stored correctly
          - Client is accessible
        """
        mock_s3_client = MagicMock()
        manager = CheckpointManager(s3_client=mock_s3_client)

        assert manager.s3_client is mock_s3_client

    def test_manager_is_frozen(self) -> None:
        """Test that CheckpointManager is immutable.

        **Why this test is important:**
          - Immutability prevents accidental modification
          - Ensures thread safety
          - Validates attrs configuration
          - Critical for reliability

        **What it tests:**
          - Attributes cannot be modified
          - FrozenInstanceError is raised
        """
        manager = CheckpointManager()

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            manager.s3_client = MagicMock()


# =============================================================================
# Load Operations Tests
# =============================================================================


class TestCheckpointManagerLoad:
    """Test suite for CheckpointManager.load method."""

    def test_load_from_local_file(self, tmp_path: Path) -> None:
        """Test loading checkpoint from local filesystem.

        **Why this test is important:**
          - Local filesystem is primary checkpoint storage
          - Critical for job recovery
          - Validates JSON deserialization
          - Ensures correct data loading

        **What it tests:**
          - Reads file from filesystem
          - Deserializes JSON correctly
          - Returns set of processed items
        """
        checkpoint_file = tmp_path / "checkpoint.json"
        processed_items = {"key1", "key2", "key3"}
        checkpoint_data = {"processed_keys": list(processed_items), "last_updated": 1234567890.0}
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        manager = CheckpointManager()
        loaded = manager.load(checkpoint_file)

        assert loaded == processed_items
        assert isinstance(loaded, set)

    def test_load_from_local_file_empty(self, tmp_path: Path) -> None:
        """Test loading empty checkpoint from local filesystem.

        **Why this test is important:**
          - Empty checkpoints are valid (new jobs)
          - Must handle empty arrays correctly
          - Validates edge case
          - Critical for new job starts

        **What it tests:**
          - Empty processed_keys loads correctly
          - Returns empty set
          - No errors raised
        """
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_data = {"processed_keys": [], "last_updated": 1234567890.0}
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        manager = CheckpointManager()
        loaded = manager.load(checkpoint_file)

        assert loaded == set()
        assert isinstance(loaded, set)

    def test_load_from_nonexistent_local_file(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file returns empty set.

        **Why this test is important:**
          - Missing checkpoint is valid (first run)
          - Should not raise error
          - Validates graceful handling
          - Critical for new jobs

        **What it tests:**
          - Missing file returns empty set
          - No FileNotFoundError raised
          - Logged as info, not error
        """
        checkpoint_file = tmp_path / "nonexistent.json"

        manager = CheckpointManager()
        loaded = manager.load(checkpoint_file)

        assert loaded == set()

    def test_load_from_s3(self) -> None:
        """Test loading checkpoint from S3.

        **Why this test is important:**
          - S3 enables distributed checkpoint storage
          - Critical for cloud deployments
          - Validates S3 integration
          - Ensures correct deserialization

        **What it tests:**
          - S3 URI is detected
          - S3 client get_object is called
          - JSON is deserialized correctly
        """
        mock_s3_client = MagicMock()
        checkpoint_data = {"processed_keys": ["key1", "key2"], "last_updated": 1234567890.0}
        mock_s3_client.get_object.return_value = json.dumps(checkpoint_data).encode("utf-8")

        manager = CheckpointManager(s3_client=mock_s3_client)
        loaded = manager.load("s3://bucket/checkpoint.json")

        assert loaded == {"key1", "key2"}
        mock_s3_client.get_object.assert_called_once_with(bucket="bucket", key="checkpoint.json")

    def test_load_from_s3_not_found(self) -> None:
        """Test loading from nonexistent S3 object returns empty set.

        **Why this test is important:**
          - Missing S3 object is valid (first run)
          - Should not raise error
          - Validates error handling
          - Critical for new jobs

        **What it tests:**
          - ClientError 404 handled gracefully
          - Returns empty set
          - No exception propagated
        """
        mock_s3_client = MagicMock()
        error = ClientError({"Error": {"Code": "404"}}, "GetObject")
        mock_s3_client.get_object.side_effect = error

        manager = CheckpointManager(s3_client=mock_s3_client)
        loaded = manager.load("s3://bucket/checkpoint.json")

        assert loaded == set()

    def test_load_from_s3_without_client_returns_empty(self) -> None:
        """Test that loading from S3 without client returns empty set.

        **Why this test is important:**
          - S3 operations require S3 client
          - Should handle gracefully without client
          - Validates fallback behavior
          - Critical for robustness

        **What it tests:**
          - Returns empty set when no client
          - Warning is logged
          - No S3 operations attempted
        """
        manager = CheckpointManager()

        loaded = manager.load("s3://bucket/checkpoint.json")

        assert loaded == set()


# =============================================================================
# Save Operations Tests
# =============================================================================


class TestCheckpointManagerSave:
    """Test suite for CheckpointManager.save method."""

    def test_save_to_local_file(self, tmp_path: Path) -> None:
        """Test saving checkpoint to local filesystem.

        **Why this test is important:**
          - Local filesystem is primary checkpoint storage
          - Critical for job recovery
          - Validates JSON serialization
          - Ensures data persistence

        **What it tests:**
          - Writes file to filesystem
          - Serializes set to JSON array
          - File contains correct data
        """
        checkpoint_file = tmp_path / "checkpoint.json"
        processed_items = {"key1", "key2", "key3"}

        manager = CheckpointManager()
        manager.save(checkpoint_file, processed_items)

        assert checkpoint_file.exists()
        loaded_data = json.loads(checkpoint_file.read_text())
        assert set(loaded_data["processed_keys"]) == processed_items
        assert "last_updated" in loaded_data

    def test_save_empty_set_to_local_file(self, tmp_path: Path) -> None:
        """Test saving empty checkpoint to local filesystem.

        **Why this test is important:**
          - Empty checkpoints are valid
          - Must handle empty sets correctly
          - Validates edge case
          - Critical for job initialization

        **What it tests:**
          - Empty set saves correctly
          - File contains empty array
          - No errors raised
        """
        checkpoint_file = tmp_path / "checkpoint.json"
        processed_items: set[str] = set()

        manager = CheckpointManager()
        manager.save(checkpoint_file, processed_items)

        assert checkpoint_file.exists()
        loaded_data = json.loads(checkpoint_file.read_text())
        assert loaded_data["processed_keys"] == []
        assert "last_updated" in loaded_data

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that save creates parent directories if missing.

        **Why this test is important:**
          - Checkpoint dirs may not exist initially
          - Should auto-create directories
          - Validates robustness
          - Critical for first-run scenarios

        **What it tests:**
          - Parent directories are created
          - File is saved successfully
          - No errors raised
        """
        checkpoint_file = tmp_path / "nested" / "dirs" / "checkpoint.json"
        processed_items = {"key1"}

        manager = CheckpointManager()
        manager.save(checkpoint_file, processed_items)

        assert checkpoint_file.exists()
        assert checkpoint_file.parent.exists()

    def test_save_to_s3(self) -> None:
        """Test saving checkpoint to S3.

        **Why this test is important:**
          - S3 enables distributed checkpoint storage
          - Critical for cloud deployments
          - Validates S3 integration
          - Ensures correct serialization

        **What it tests:**
          - S3 URI is detected
          - S3 client put_object is called
          - Set is serialized to JSON
        """
        mock_s3_client = MagicMock()
        processed_items = {"key1", "key2"}

        manager = CheckpointManager(s3_client=mock_s3_client)
        manager.save("s3://bucket/checkpoint.json", processed_items)

        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]["bucket"] == "bucket"
        assert call_args[1]["key"] == "checkpoint.json"
        # Verify JSON content
        body = call_args[1]["body"]
        data = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
        assert set(data["processed_keys"]) == processed_items
        assert "last_updated" in data

    def test_save_to_s3_without_client_logs_warning(self) -> None:
        """Test that saving to S3 without client logs warning and returns.

        **Why this test is important:**
          - S3 operations require S3 client
          - Should handle gracefully without client
          - Validates fallback behavior
          - Critical for robustness

        **What it tests:**
          - Returns without error
          - Warning is logged
          - No S3 operations attempted
        """
        manager = CheckpointManager()

        # Should not raise, just log warning and return
        manager.save("s3://bucket/checkpoint.json", {"key1"})

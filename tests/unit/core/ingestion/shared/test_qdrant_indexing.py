"""Unit tests for core.ingestion.shared.qdrant_indexing module.

Tests the qdrant_indexing_disabled context manager that temporarily disables
HNSW indexing on a Qdrant collection for faster bulk uploads.

Run with: uv run pytest tests/unit/core/ingestion/shared/test_qdrant_indexing.py -v
"""

import logging
from unittest.mock import MagicMock

import pytest

from core.ingestion.shared.qdrant_indexing import qdrant_indexing_disabled


def _make_client(*, hnsw_m: int = 16, indexing_threshold: int = 20_000) -> MagicMock:
    """Create a mock QdrantClientWrapper whose disable_indexing_sync returns
    a CollectionConfig with the given HNSW and optimizer parameters."""
    client = MagicMock()
    config = MagicMock()
    config.hnsw_config.m = hnsw_m
    config.optimizer_config.indexing_threshold = indexing_threshold
    client.disable_indexing_sync.return_value = config
    return client


class TestQdrantIndexingDisabled:
    """Test suite for qdrant_indexing_disabled context manager."""

    def test_happy_path_disables_and_reenables(self):
        """Indexing is disabled on entry and re-enabled on exit with original params.

        **What it tests:**
          - disable_indexing_sync is called with the correct collection and vector_size
          - enable_indexing_sync is called with the original m and indexing_threshold
          - The body of the with block executes
        """
        client = _make_client(hnsw_m=16, indexing_threshold=20_000)
        body_ran = False

        with qdrant_indexing_disabled(client, "my-collection", vector_size=512):
            body_ran = True

        assert body_ran
        client.disable_indexing_sync.assert_called_once_with(collection="my-collection", vector_size=512)
        client.enable_indexing_sync.assert_called_once_with(
            collection="my-collection", indexing_threshold=20_000, hnsw_m=16
        )

    def test_disable_fails_body_still_runs_enable_skipped(self, caplog):
        """When disable_indexing_sync raises, the body still executes and
        enable_indexing_sync is not called.

        **What it tests:**
          - The with-block body runs even if disabling fails
          - enable_indexing_sync is never called (index_disabled stays False)
          - The "Indexing was already enabled" info log is emitted
        """
        client = MagicMock()
        client.disable_indexing_sync.side_effect = RuntimeError("connection refused")
        body_ran = False

        with caplog.at_level(logging.INFO, logger="pipeline.ingestion.qdrant_indexing"):
            with qdrant_indexing_disabled(client, "col", vector_size=256):
                body_ran = True

        assert body_ran
        client.enable_indexing_sync.assert_not_called()
        assert any("Indexing was already enabled" in r.message for r in caplog.records)

    def test_enable_fails_exception_swallowed(self, caplog):
        """When enable_indexing_sync raises, the exception is swallowed and logged.

        **What it tests:**
          - No exception propagates out of the context manager
          - The "Failed to enable Qdrant indexing" error log is emitted
        """
        client = _make_client()
        client.enable_indexing_sync.side_effect = RuntimeError("timeout")

        with caplog.at_level(logging.ERROR, logger="pipeline.ingestion.qdrant_indexing"):
            with qdrant_indexing_disabled(client, "col", vector_size=512):
                pass

        assert any("Failed to enable Qdrant indexing" in r.message for r in caplog.records)

    def test_body_raises_indexing_still_reenabled(self):
        """When the with-block body raises, indexing is still re-enabled and
        the original exception propagates.

        **What it tests:**
          - enable_indexing_sync is called even when the body raises
          - The RuntimeError from the body propagates out
        """
        client = _make_client()

        with pytest.raises(RuntimeError, match="boom"):
            with qdrant_indexing_disabled(client, "col", vector_size=512):
                raise RuntimeError("boom")

        client.enable_indexing_sync.assert_called_once()

    def test_original_params_forwarded_correctly(self):
        """Unusual original config values are forwarded to enable_indexing_sync.

        **What it tests:**
          - Non-default m and indexing_threshold values from the original config
            are passed through to enable_indexing_sync exactly
        """
        client = _make_client(hnsw_m=32, indexing_threshold=50_000)

        with qdrant_indexing_disabled(client, "special-col", vector_size=1024):
            pass

        client.enable_indexing_sync.assert_called_once_with(
            collection="special-col", indexing_threshold=50_000, hnsw_m=32
        )

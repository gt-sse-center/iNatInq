"""Unit tests for core.ingestion.shared.qdrant_indexing module.

Tests the disable/enable Qdrant indexing helpers that use the synchronous
QdrantClient for bulk ingestion optimisation.

Run with: uv run pytest tests/unit/core/ingestion/shared/test_qdrant_indexing.py -v
"""

import contextlib
from unittest.mock import MagicMock, patch

from qdrant_client.http import models as qmodels

from core.ingestion.shared.qdrant_indexing import disable_qdrant_indexing, enable_qdrant_indexing


class TestDisableQdrantIndexing:
    """Tests for disable_qdrant_indexing."""

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_calls_update_collection_with_zero_thresholds(self, mock_client_cls: MagicMock) -> None:
        """Verify update_collection is called with indexing_threshold=0 and m=0."""
        mock_client = mock_client_cls.return_value

        disable_qdrant_indexing("http://localhost:6333", None, "my_collection_images")

        mock_client.update_collection.assert_called_once_with(
            collection_name="my_collection_images",
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=0),
            hnsw_config=qmodels.HnswConfigDiff(m=0),
        )

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_creates_client_with_url_and_api_key(self, mock_client_cls: MagicMock) -> None:
        """Verify QdrantClient is constructed with the correct url and api_key."""
        disable_qdrant_indexing("http://qdrant:6333", "secret-key", "col")

        mock_client_cls.assert_called_once_with(url="http://qdrant:6333", api_key="secret-key")

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_closes_client_after_success(self, mock_client_cls: MagicMock) -> None:
        """Verify the sync client is closed after a successful disable call."""
        mock_client = mock_client_cls.return_value

        disable_qdrant_indexing("http://localhost:6333", None, "col")

        mock_client.close.assert_called_once()

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_closes_client_on_error(self, mock_client_cls: MagicMock) -> None:
        """Verify the sync client is closed even when update_collection raises."""
        mock_client = mock_client_cls.return_value
        mock_client.update_collection.side_effect = RuntimeError("boom")

        with contextlib.suppress(RuntimeError):
            disable_qdrant_indexing("http://localhost:6333", None, "col")

        mock_client.close.assert_called_once()


class TestEnableQdrantIndexing:
    """Tests for enable_qdrant_indexing."""

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_calls_update_collection_with_defaults(self, mock_client_cls: MagicMock) -> None:
        """Verify update_collection restores default indexing_threshold and hnsw_m."""
        mock_client = mock_client_cls.return_value

        enable_qdrant_indexing("http://localhost:6333", None, "my_collection_images")

        mock_client.update_collection.assert_called_once_with(
            collection_name="my_collection_images",
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=20_000),
            hnsw_config=qmodels.HnswConfigDiff(m=16),
        )

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_calls_update_collection_with_custom_params(self, mock_client_cls: MagicMock) -> None:
        """Verify custom indexing_threshold and hnsw_m are forwarded."""
        mock_client = mock_client_cls.return_value

        enable_qdrant_indexing(
            "http://localhost:6333",
            None,
            "col",
            indexing_threshold=50_000,
            hnsw_m=32,
        )

        mock_client.update_collection.assert_called_once_with(
            collection_name="col",
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=50_000),
            hnsw_config=qmodels.HnswConfigDiff(m=32),
        )

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_creates_client_with_url_and_api_key(self, mock_client_cls: MagicMock) -> None:
        """Verify QdrantClient is constructed with the correct url and api_key."""
        enable_qdrant_indexing("http://qdrant:6333", "my-key", "col")

        mock_client_cls.assert_called_once_with(url="http://qdrant:6333", api_key="my-key")

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_closes_client_after_success(self, mock_client_cls: MagicMock) -> None:
        """Verify the sync client is closed after a successful enable call."""
        mock_client = mock_client_cls.return_value

        enable_qdrant_indexing("http://localhost:6333", None, "col")

        mock_client.close.assert_called_once()

    @patch("core.ingestion.shared.qdrant_indexing.QdrantClient")
    def test_closes_client_on_error(self, mock_client_cls: MagicMock) -> None:
        """Verify the sync client is closed even when update_collection raises."""
        mock_client = mock_client_cls.return_value
        mock_client.update_collection.side_effect = RuntimeError("boom")

        with contextlib.suppress(RuntimeError):
            enable_qdrant_indexing("http://localhost:6333", None, "col")

        mock_client.close.assert_called_once()

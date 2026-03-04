"""Unit tests for benchmark provider factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.benchmark.provider_factory import resolve_search_pipeline


class TestResolveSearchPipeline:
    """Tests for resolve_search_pipeline."""

    def test_qdrant_provider_creates_pipeline(self, mocker):
        """resolve_search_pipeline for 'qdrant' creates a CLIPSearchPipeline."""
        mock_vdb_config_cls = mocker.patch("core.benchmark.provider_factory.VectorDBConfig")
        mock_emb_config_cls = mocker.patch("core.benchmark.provider_factory.EmbeddingConfig")
        mock_clip_cls = mocker.patch("core.benchmark.provider_factory.CLIPClient")
        mock_create_vdb = mocker.patch("core.benchmark.provider_factory.create_vector_db_provider")

        mock_vdb_config = MagicMock()
        mock_vdb_config_cls.from_env_for_provider.return_value = mock_vdb_config
        mock_provider = MagicMock()
        mock_create_vdb.return_value = mock_provider

        mock_emb_config = MagicMock()
        mock_emb_config_cls.from_env.return_value = mock_emb_config
        mock_clip_client = MagicMock()
        mock_clip_cls.from_config.return_value = mock_clip_client

        provider, pipeline = resolve_search_pipeline("qdrant")

        assert provider is mock_provider
        assert pipeline.clip_client is mock_clip_client
        assert pipeline.vector_provider is mock_provider

    def test_qdrant_passes_collection_to_config(self, mocker):
        """Collection name is passed to VectorDBConfig."""
        mock_vdb_config_cls = mocker.patch("core.benchmark.provider_factory.VectorDBConfig")
        mock_emb_config_cls = mocker.patch("core.benchmark.provider_factory.EmbeddingConfig")
        mock_clip_cls = mocker.patch("core.benchmark.provider_factory.CLIPClient")
        mock_create_vdb = mocker.patch("core.benchmark.provider_factory.create_vector_db_provider")

        mock_vdb_config = MagicMock()
        mock_vdb_config_cls.from_env_for_provider.return_value = mock_vdb_config
        mock_create_vdb.return_value = MagicMock()
        mock_emb_config_cls.from_env.return_value = MagicMock()
        mock_clip_cls.from_config.return_value = MagicMock()

        resolve_search_pipeline("qdrant", collection="my-collection")

        mock_vdb_config_cls.from_env_for_provider.assert_called_once_with("qdrant")

    def test_unknown_provider_raises(self):
        """Unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_search_pipeline("unknown_db")

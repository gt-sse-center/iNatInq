"""Unit tests for search service Prometheus metrics.

Verifies that `ImageSearchService` emits the correct histogram observations
for embedding duration, vector query duration, and result count on both
the happy path and error paths.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY

from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from core.models import SearchResultItem, SearchResults
from core.services.search_service import ImageSearchService
from foundation.exceptions import UpstreamError


def _histogram_count(metric_name: str, labels: dict[str, str]) -> float:
    """Return the `_count` sample for a histogram label-set, or 0.0 if absent."""
    return REGISTRY.get_sample_value(f"{metric_name}_count", labels) or 0.0


@pytest.fixture
def embedding_provider() -> MagicMock:
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return provider


@pytest.fixture
def vector_db_provider() -> MagicMock:
    provider = MagicMock(spec=VectorDBProvider)
    provider.search_async = AsyncMock(
        return_value=SearchResults(
            items=[
                SearchResultItem(point_id="1", score=0.9, payload={}),
                SearchResultItem(point_id="2", score=0.8, payload={}),
                SearchResultItem(point_id="3", score=0.7, payload={}),
            ],
            total=3,
        )
    )
    return provider


@pytest.fixture
def service(embedding_provider: MagicMock, vector_db_provider: MagicMock) -> ImageSearchService:
    return ImageSearchService(
        embedding_provider=embedding_provider,
        vector_db_provider=vector_db_provider,
        embedding_provider_name="clip",
        vector_db_provider_name="qdrant",
    )


class TestSearchEmbeddingDuration:
    """inatinq_search_embedding_duration_seconds is observed on each search."""

    @pytest.mark.asyncio
    async def test_search_records_embedding_duration(self, service: ImageSearchService) -> None:
        """Embedding histogram is observed (count +1) with the correct provider label."""
        labels = {"provider": "clip"}
        before_count = _histogram_count("inatinq_search_embedding_duration_seconds", labels)
        before_sum = REGISTRY.get_sample_value("inatinq_search_embedding_duration_seconds_sum", labels) or 0.0
        await service.search_images_async(collection="docs", query="cat", limit=5)
        after_count = _histogram_count("inatinq_search_embedding_duration_seconds", labels)
        after_sum = REGISTRY.get_sample_value("inatinq_search_embedding_duration_seconds_sum", labels)
        assert after_count == before_count + 1.0
        assert after_sum is not None
        assert after_sum > before_sum


class TestSearchVectorQueryDuration:
    """inatinq_search_vector_query_duration_seconds is observed on each search."""

    @pytest.mark.asyncio
    async def test_search_records_vector_query_duration(self, service: ImageSearchService) -> None:
        """Vector query histogram is observed (count +1) with correct provider and collection labels."""
        labels = {"provider": "qdrant", "collection": "photos"}
        before_count = _histogram_count("inatinq_search_vector_query_duration_seconds", labels)
        before_sum = (
            REGISTRY.get_sample_value("inatinq_search_vector_query_duration_seconds_sum", labels) or 0.0
        )
        await service.search_images_async(collection="photos", query="sunset", limit=5)
        after_count = _histogram_count("inatinq_search_vector_query_duration_seconds", labels)
        after_sum = REGISTRY.get_sample_value("inatinq_search_vector_query_duration_seconds_sum", labels)
        assert after_count == before_count + 1.0
        assert after_sum is not None
        assert after_sum > before_sum


class TestSearchResultCount:
    """inatinq_search_result_count is observed with the number of results returned."""

    @pytest.mark.asyncio
    async def test_search_records_result_count(self, service: ImageSearchService) -> None:
        """Result count histogram count increments by 1 after a successful search."""
        labels = {"collection": "wildlife"}
        before = _histogram_count("inatinq_search_result_count", labels)
        await service.search_images_async(collection="wildlife", query="bird", limit=10)
        after = _histogram_count("inatinq_search_result_count", labels)
        assert after == before + 1.0


class TestImageSearchProviderLabel:
    """Provider label reflects the name configured on the service."""

    @pytest.mark.asyncio
    async def test_image_search_uses_clip_label(
        self,
        embedding_provider: MagicMock,
        vector_db_provider: MagicMock,
    ) -> None:
        """Embedding metric carries provider=clip when service is configured with clip."""
        clip_labels = {"provider": "clip"}
        ollama_labels = {"provider": "ollama"}
        before_clip = _histogram_count("inatinq_search_embedding_duration_seconds", clip_labels)
        before_ollama = _histogram_count("inatinq_search_embedding_duration_seconds", ollama_labels)
        svc = ImageSearchService(
            embedding_provider=embedding_provider,
            vector_db_provider=vector_db_provider,
            embedding_provider_name="clip",
            vector_db_provider_name="qdrant",
        )
        await svc.search_images_async(collection="docs", query="bird", limit=5)
        after_clip = _histogram_count("inatinq_search_embedding_duration_seconds", clip_labels)
        after_ollama = _histogram_count("inatinq_search_embedding_duration_seconds", ollama_labels)
        assert after_clip == before_clip + 1.0
        assert after_ollama == before_ollama


class TestMetricsEmittedOnError:
    """Embedding duration is still observed even when the vector query fails."""

    @pytest.mark.asyncio
    async def test_metrics_emitted_on_error(
        self,
        embedding_provider: MagicMock,
        vector_db_provider: MagicMock,
    ) -> None:
        """Embedding and vector query histograms are observed before re-raising a vector DB error."""
        vector_db_provider.search_async.side_effect = UpstreamError("Qdrant down")
        svc = ImageSearchService(
            embedding_provider=embedding_provider,
            vector_db_provider=vector_db_provider,
            embedding_provider_name="clip",
            vector_db_provider_name="qdrant",
        )
        embed_labels = {"provider": "clip"}
        vq_labels = {"provider": "qdrant", "collection": "docs"}
        before_embed_count = _histogram_count("inatinq_search_embedding_duration_seconds", embed_labels)
        before_vq_count = _histogram_count("inatinq_search_vector_query_duration_seconds", vq_labels)
        before_embed_sum = (
            REGISTRY.get_sample_value("inatinq_search_embedding_duration_seconds_sum", embed_labels) or 0.0
        )
        before_vq_sum = (
            REGISTRY.get_sample_value("inatinq_search_vector_query_duration_seconds_sum", vq_labels) or 0.0
        )

        with pytest.raises(UpstreamError, match="Qdrant down"):
            await svc.search_images_async(collection="docs", query="cat", limit=5)

        after_embed_count = _histogram_count("inatinq_search_embedding_duration_seconds", embed_labels)
        after_vq_count = _histogram_count("inatinq_search_vector_query_duration_seconds", vq_labels)
        after_embed_sum = REGISTRY.get_sample_value(
            "inatinq_search_embedding_duration_seconds_sum", embed_labels
        )
        after_vq_sum = REGISTRY.get_sample_value(
            "inatinq_search_vector_query_duration_seconds_sum", vq_labels
        )
        assert after_embed_count == before_embed_count + 1.0
        assert after_embed_sum is not None
        assert after_embed_sum > before_embed_sum
        assert after_vq_count == before_vq_count + 1.0
        assert after_vq_sum is not None
        assert after_vq_sum > before_vq_sum

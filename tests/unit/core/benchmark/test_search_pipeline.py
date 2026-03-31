"""Unit tests for benchmark search pipeline.

Tests search pipeline adapter for benchmarks, result ID extraction, and error
handling during query execution. This is important because the search pipeline
adapter bridges the benchmark framework to the actual search implementation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from qdrant_client.http import models as qmodels

from clients.qdrant import QdrantClientWrapper
from core.benchmark.id_mapping import S3KeyIDMapper
from core.benchmark.search_pipeline import QdrantSearchPipeline, SearchPipeline, SearchPipelineResult
from core.models import SearchResultItem, SearchResults


def _make_search_results(*items: tuple[str, dict]) -> SearchResults:
    """Build SearchResults with point_id and payload tuples."""
    result_items = [
        SearchResultItem(point_id=pid, score=1.0 - i * 0.1, payload=payload)
        for i, (pid, payload) in enumerate(items)
    ]
    return SearchResults(items=result_items, total=len(result_items))


class TestSearchPipelineResult:
    """Tests for SearchPipelineResult data class."""

    def test_construction(self):
        """SearchPipelineResult holds doc_ids and raw_results."""
        raw = _make_search_results(("uuid1", {"s3_key": "1"}))
        result = SearchPipelineResult(doc_ids=["1"], raw_results=raw)
        assert result.doc_ids == ["1"]
        assert result.raw_results is raw


class TestSearchPipeline:
    """Tests for SearchPipeline."""

    @pytest.mark.asyncio
    async def test_search_embed_then_search_then_map(self):
        """Pipeline calls embed_text (async), then search_async, then maps IDs."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.1, 0.2, 0.3]

        vector_provider = AsyncMock()
        vector_provider.search_async.return_value = _make_search_results(
            ("uuid-a", {"s3_key": "100"}),
            ("uuid-b", {"s3_key": "200"}),
        )

        mapper = S3KeyIDMapper()
        pipeline = SearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=vector_provider,
            id_mapper=mapper,
        )

        result = await pipeline.search("red bird", collection="test-col", limit=10)

        embedding_provider.embed_text.assert_called_once_with("red bird")
        vector_provider.search_async.assert_called_once_with(
            collection="test-col",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
        )
        assert isinstance(result, SearchPipelineResult)
        assert result.doc_ids == ["100", "200"]

    @pytest.mark.asyncio
    async def test_search_falls_back_to_point_id_without_s3_key(self):
        """When payload lacks s3_key, doc_id falls back to point_id."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.1]

        vector_provider = AsyncMock()
        vector_provider.search_async.return_value = _make_search_results(
            ("uuid-x", {}),
        )

        mapper = S3KeyIDMapper()
        pipeline = SearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=vector_provider,
            id_mapper=mapper,
        )

        result = await pipeline.search("query", collection="col", limit=5)
        assert result.doc_ids == ["uuid-x"]

    @pytest.mark.asyncio
    async def test_search_preserves_raw_results(self):
        """Raw SearchResults are preserved in the pipeline result."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.5]

        raw = _make_search_results(("uuid-1", {"s3_key": "42"}))
        vector_provider = AsyncMock()
        vector_provider.search_async.return_value = raw

        pipeline = SearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=vector_provider,
            id_mapper=S3KeyIDMapper(),
        )

        result = await pipeline.search("test", collection="c", limit=1)
        assert result.raw_results is raw

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Pipeline handles empty search results."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.1]

        vector_provider = AsyncMock()
        vector_provider.search_async.return_value = SearchResults(items=[], total=0)

        pipeline = SearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=vector_provider,
            id_mapper=S3KeyIDMapper(),
        )

        result = await pipeline.search("query", collection="col", limit=10)
        assert result.doc_ids == []
        assert len(result.raw_results) == 0


class TestQdrantSearchPipeline:
    """Tests for QdrantSearchPipeline with Qdrant-specific search_params."""

    @pytest.mark.asyncio
    async def test_passes_search_params_to_qdrant_client(self):
        """QdrantSearchPipeline forwards search_params to QdrantClientWrapper.search_async."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.1, 0.2, 0.3]

        # spec=QdrantClientWrapper makes isinstance() checks pass
        mock_qdrant = AsyncMock(spec=QdrantClientWrapper)
        mock_qdrant.search_async.return_value = _make_search_results(
            ("uuid-a", {"s3_key": "100"}),
        )

        search_params = qmodels.SearchParams(
            quantization=qmodels.QuantizationSearchParams(rescore=True, oversampling=3.0)
        )

        pipeline = QdrantSearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=mock_qdrant,
            id_mapper=S3KeyIDMapper(),
            search_params=search_params,
        )

        result = await pipeline.search("red bird", collection="test-col", limit=10)

        mock_qdrant.search_async.assert_called_once_with(
            collection="test-col",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            search_params=search_params,
        )
        assert result.doc_ids == ["100"]

    @pytest.mark.asyncio
    async def test_default_search_params_is_none(self):
        """QdrantSearchPipeline defaults search_params to None."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.5]

        mock_qdrant = AsyncMock(spec=QdrantClientWrapper)
        mock_qdrant.search_async.return_value = _make_search_results(
            ("uuid-b", {"s3_key": "200"}),
        )

        pipeline = QdrantSearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=mock_qdrant,
            id_mapper=S3KeyIDMapper(),
        )

        result = await pipeline.search("query", collection="col", limit=5)

        mock_qdrant.search_async.assert_called_once_with(
            collection="col",
            query_vector=[0.5],
            limit=5,
            search_params=None,
        )
        assert result.doc_ids == ["200"]

    @pytest.mark.asyncio
    async def test_raises_type_error_for_non_qdrant_provider(self):
        """QdrantSearchPipeline raises TypeError for non-Qdrant providers."""
        embedding_provider = AsyncMock()
        embedding_provider.embed_text.return_value = [0.1]

        non_qdrant_provider = AsyncMock()

        pipeline = QdrantSearchPipeline(
            embedding_provider=embedding_provider,
            vector_provider=non_qdrant_provider,
            id_mapper=S3KeyIDMapper(),
        )

        with pytest.raises(TypeError, match="QdrantSearchPipeline requires QdrantClientWrapper"):
            await pipeline.search("query", collection="col", limit=5)

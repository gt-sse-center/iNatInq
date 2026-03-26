"""Resolve provider names to client instances for benchmarking.

Given a provider name (e.g., ``"qdrant"``), this module constructs the
full search pipeline: VectorDBProvider + EmbeddingProvider + S3KeyIDMapper →
SearchPipeline.

Example:
    ```python
    from core.benchmark.provider_factory import resolve_search_pipeline

    provider, pipeline = resolve_search_pipeline("qdrant")
    result = await pipeline.search("red bird", collection="images", limit=10)
    ```

Quantization profiles can be passed to create a ``QdrantSearchPipeline``
with Qdrant-specific ``SearchParams``::

    provider, pipeline = resolve_search_pipeline(
        "qdrant", quantization_profile="scalar-rescore",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from clients.interfaces.embedding import create_embedding_provider
from clients.interfaces.vector_db import VectorDBProvider, create_vector_db_provider
from config import EmbeddingConfig, VectorDBConfig
from core.benchmark.id_mapping import S3KeyIDMapper
from core.benchmark.search_pipeline import QdrantSearchPipeline, SearchPipeline

logger = logging.getLogger("benchmark.provider_factory")

_SUPPORTED_PROVIDERS = {"qdrant"}

_QUANTIZATION_PROFILES: dict[str, Any] = {}


def _ensure_profiles_loaded() -> None:
    """Lazily build the quantization profile registry.

    Deferred so ``qdrant_client`` is only imported when actually needed.
    """
    if _QUANTIZATION_PROFILES:
        return

    from qdrant_client.http import models as qmodels

    _QUANTIZATION_PROFILES.update(
        {
            "none": None,
            "scalar": None,
            "scalar-rescore": qmodels.SearchParams(
                quantization=qmodels.QuantizationSearchParams(rescore=True),
            ),
            "binary-rescore": qmodels.SearchParams(
                quantization=qmodels.QuantizationSearchParams(
                    rescore=True,
                    oversampling=3.0,
                ),
            ),
        }
    )


def list_quantization_profiles() -> list[str]:
    """Return the names of all registered quantization profiles."""
    _ensure_profiles_loaded()
    return sorted(_QUANTIZATION_PROFILES.keys())


def resolve_search_pipeline(
    provider_name: str,
    *,
    collection: str | None = None,  # noqa: ARG001 — passed through by caller, not used here
    quantization_profile: str | None = None,
) -> tuple[VectorDBProvider, SearchPipeline]:
    """Resolve a provider name to a VectorDBProvider and SearchPipeline.

    Constructs the full pipeline from environment configuration:
    1. VectorDBConfig → VectorDBProvider (via factory + registry)
    2. EmbeddingConfig → EmbeddingProvider (via factory + registry)
    3. S3KeyIDMapper for ID translation
    4. Compose into SearchPipeline (or QdrantSearchPipeline for quantization)

    Args:
        provider_name: Provider identifier (e.g., ``"qdrant"``).
        collection: Optional collection name override. Passed through
            but not used for provider construction (the runner handles it).
        quantization_profile: Optional quantization profile name. When set
            and the provider is ``"qdrant"``, a ``QdrantSearchPipeline`` is
            created with the corresponding ``SearchParams``. Use
            ``list_quantization_profiles()`` to see available names.

    Returns:
        Tuple of (VectorDBProvider, SearchPipeline).

    Raises:
        ValueError: If provider_name or quantization_profile is not recognized.
    """
    if provider_name not in _SUPPORTED_PROVIDERS:
        msg = f"Unknown provider '{provider_name}'. Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        raise ValueError(msg)

    # Ensure providers are registered
    import clients.registries  # noqa: F401

    # Build vector DB provider from env config
    vdb_config = VectorDBConfig.from_env_for_provider(provider_name)
    vector_provider = create_vector_db_provider(vdb_config)

    # Build embedding provider from env config
    emb_config = EmbeddingConfig.from_env()
    image_provider = create_embedding_provider(emb_config)

    # Compose the pipeline
    id_mapper = S3KeyIDMapper()
    pipeline: SearchPipeline

    if quantization_profile is not None and provider_name == "qdrant":
        _ensure_profiles_loaded()
        if quantization_profile not in _QUANTIZATION_PROFILES:
            msg = (
                f"Unknown quantization profile '{quantization_profile}'. "
                f"Available: {list_quantization_profiles()}"
            )
            raise ValueError(msg)

        search_params = _QUANTIZATION_PROFILES[quantization_profile]
        if search_params is not None:
            pipeline = QdrantSearchPipeline(
                embedding_provider=image_provider,
                vector_provider=vector_provider,
                id_mapper=id_mapper,
                search_params=search_params,
            )
        else:
            pipeline = SearchPipeline(
                embedding_provider=image_provider,
                vector_provider=vector_provider,
                id_mapper=id_mapper,
            )
    else:
        pipeline = SearchPipeline(
            embedding_provider=image_provider,
            vector_provider=vector_provider,
            id_mapper=id_mapper,
        )

    logger.info(
        "Resolved search pipeline",
        extra={
            "provider": provider_name,
            "clip_url": emb_config.clip_url,
            "quantization_profile": quantization_profile,
        },
    )

    return vector_provider, pipeline

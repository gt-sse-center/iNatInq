"""Resolve provider names to client instances for benchmarking.

Given a provider name (e.g., ``"qdrant"``), this module constructs the
full search pipeline: VectorDBProvider + CLIPClient + S3KeyIDMapper →
CLIPSearchPipeline.

Example:
    ```python
    from core.benchmark.provider_factory import resolve_search_pipeline

    provider, pipeline = resolve_search_pipeline("qdrant")
    result = await pipeline.search("red bird", collection="images", limit=10)
    ```
"""

from __future__ import annotations

import logging

from clients.clip import CLIPClient
from clients.interfaces.vector_db import VectorDBProvider, create_vector_db_provider
from config import ImageEmbeddingConfig, VectorDBConfig
from core.benchmark.id_mapping import S3KeyIDMapper
from core.benchmark.search_pipeline import CLIPSearchPipeline

logger = logging.getLogger("benchmark.provider_factory")

_SUPPORTED_PROVIDERS = {"qdrant", "weaviate"}


def resolve_search_pipeline(
    provider_name: str,
    *,
    collection: str | None = None,  # noqa: ARG001 — passed through by caller, not used here
) -> tuple[VectorDBProvider, CLIPSearchPipeline]:
    """Resolve a provider name to a VectorDBProvider and CLIPSearchPipeline.

    Constructs the full pipeline from environment configuration:
    1. VectorDBConfig → VectorDBProvider (via factory + registry)
    2. ImageEmbeddingConfig → CLIPClient
    3. S3KeyIDMapper for ID translation
    4. Compose into CLIPSearchPipeline

    Args:
        provider_name: Provider identifier (e.g., ``"qdrant"``).
        collection: Optional collection name override. Passed through
            but not used for provider construction (the runner handles it).

    Returns:
        Tuple of (VectorDBProvider, CLIPSearchPipeline).

    Raises:
        ValueError: If provider_name is not recognized.
    """
    if provider_name not in _SUPPORTED_PROVIDERS:
        msg = f"Unknown provider '{provider_name}'. Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        raise ValueError(msg)

    # Ensure providers are registered
    import clients.registries  # noqa: F401

    # Build vector DB provider from env config
    vdb_config = VectorDBConfig.from_env_for_provider(provider_name)
    vector_provider = create_vector_db_provider(vdb_config)

    # Build CLIP client from env config
    img_config = ImageEmbeddingConfig.from_env()
    clip_client = CLIPClient.from_config(img_config)

    # Compose the pipeline
    id_mapper = S3KeyIDMapper()
    pipeline = CLIPSearchPipeline(
        clip_client=clip_client,
        vector_provider=vector_provider,
        id_mapper=id_mapper,
    )

    logger.info(
        "Resolved search pipeline",
        extra={
            "provider": provider_name,
            "clip_backend": img_config.clip_backend,
            "clip_url": img_config.clip_url,
        },
    )

    return vector_provider, pipeline

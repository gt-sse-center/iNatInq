"""Abstract interfaces and shared components for ingestion pipelines.

This package provides:
- Type classes (ProcessingResult, ContentResult, etc.)
- Factory classes (VectorDBConfigFactory, ProcessingClientsFactory, etc.)
- Operation classes (S3ContentFetcher, EmbeddingGenerator, etc.)
- Abstract base class (ProcessingPipeline)

These are shared between Ray and Spark implementations to avoid code
duplication and ensure consistent behavior.

Example:
    >>> from core.ingestion.interfaces import (
    ...     ProcessingConfig,
    ...     ProcessingResult,
    ...     ProcessingClientsFactory,
    ...     S3ContentFetcher,
    ... )
    >>>
    >>> config = ProcessingConfig(...)
    >>> clients = ProcessingClientsFactory().create(config)
    >>> fetcher = S3ContentFetcher(clients.s3, config.s3_bucket)
    >>> content = fetcher.fetch_one("inputs/doc.txt")
"""

from .factories import (
    ProcessingClientsFactory,
    VectorDBConfigFactory,
    VectorPointFactory,
)
from .operations import (
    BatchProcessor,
    EmbeddingGenerator,
    S3ContentFetcher,
    VectorDBUpserter,
)
from .pipeline import BatchProcessingPipeline, ProcessingPipeline
from .types import (
    BatchEmbeddingResult,
    ContentResult,
    ProcessingClients,
    ProcessingConfig,
    ProcessingResult,
    RateLimitConfig,
)

__all__ = [
    "BatchEmbeddingResult",
    "BatchProcessingPipeline",
    "BatchProcessor",
    "ContentResult",
    "EmbeddingGenerator",
    "ProcessingClients",
    "ProcessingClientsFactory",
    "ProcessingConfig",
    "ProcessingPipeline",
    "ProcessingResult",
    "RateLimitConfig",
    "S3ContentFetcher",
    "VectorDBConfigFactory",
    "VectorDBUpserter",
    "VectorPointFactory",
]

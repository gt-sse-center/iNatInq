"""Abstract base class for processing pipelines.

This module defines the ProcessingPipeline ABC that both Ray and Spark
implementations must conform to. It provides a common interface for
processing S3 objects through embedding and vector DB upsert.
"""

from abc import ABC, abstractmethod

from .types import ProcessingClients, ProcessingConfig, ProcessingResult


class ProcessingPipeline(ABC):
    """Abstract base class for S3-to-vector-DB processing pipelines.

    This class defines the contract that all processing implementations
    (Ray, Spark, etc.) must follow. Implementations handle the specifics
    of distributed execution while using shared operation classes.

    Example:
        >>> class RayPipeline(ProcessingPipeline):
        ...     async def process_keys(self, keys):
        ...         # Ray-specific implementation
        ...         pass
        ...
        ...     def create_clients(self):
        ...         # Create clients for Ray
        ...         pass
    """

    @abstractmethod
    async def process_keys(self, keys: list[str]) -> list[ProcessingResult]:
        """Process a list of S3 keys through the pipeline.

        This is the main entry point for processing. Implementations should:
        1. Fetch S3 content
        2. Generate embeddings
        3. Upsert to vector databases
        4. Return results for each key

        Args:
            keys: List of S3 object keys to process.

        Returns:
            List of ProcessingResult objects, one per input key.
        """

    @abstractmethod
    def create_clients(self) -> ProcessingClients:
        """Create the client bundle for this pipeline.

        Returns:
            ProcessingClients with all required external service clients.
        """

    @property
    @abstractmethod
    def config(self) -> ProcessingConfig:
        """Get the processing configuration.

        Returns:
            ProcessingConfig for this pipeline.
        """


class BatchProcessingPipeline(ProcessingPipeline):
    """Extended ABC for pipelines that support batch processing.

    Adds methods for processing batches of keys, which is the common
    pattern for both Ray and Spark implementations.
    """

    @abstractmethod
    async def process_batch(
        self,
        keys: list[str],
        batch_size: int,
    ) -> list[ProcessingResult]:
        """Process keys in batches.

        Args:
            keys: List of S3 object keys to process.
            batch_size: Number of keys per batch.

        Returns:
            List of ProcessingResult objects.
        """

    @abstractmethod
    def get_batch_size(self) -> int:
        """Get the current batch size.

        Returns:
            Current batch size for processing.
        """


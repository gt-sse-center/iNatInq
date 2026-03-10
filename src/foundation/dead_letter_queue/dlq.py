"""Provides DLQ object for registering failed ingestion jobs to the dead letter queue."""

import logging
from foundation.dead_letter_queue.dlq_backend import DLQBackend

logger = logging.getLogger("foundation.dead_letter_queue")


class DLQ:
    """Provides functionality to insert failed jobs into the dead letter queue.

    This class is automatically injected to functions wrapped with the `@with_dlq` decorator.

    Example:
        ```
        @with_dlq
        def foo(dlq: DLQ):
            dlq.enqueue_failed_ingestion("image-1", metadata={})
        ```
    """

    def __init__(self, backend: DLQBackend) -> None:
        self._backend = backend

    def enqueue_failed_ingestion(self, image_id: str, *, metadata: dict[str, object] | None = None):
        """Insert a failed image ingestion into the DLQ.

        Args:
            image_id: ID of the failed image ingestion
            metadata: optional metadata to store alongside the image_id
        """
        try:
            self._backend.insert(image_id, metadata=metadata)
        except Exception:
            logger.exception(
                "Failed to insert %s into dead letter queue",
                image_id,
                extra={"image_id": image_id, "dlq_backend": self._backend.name, "metadata": metadata},
            )

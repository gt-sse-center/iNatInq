"""DLQBackend interface - defines required behavior of objects serving as a dead letter queue backend."""

from abc import ABC, abstractmethod
from typing_extensions import override


class DLQBackend(ABC):
    """Interface defining expected behavior for a DLQ backend."""

    @abstractmethod
    def insert(self, image_id: str, *, metadata: dict[str, object] | None = None):
        """Insert ID of a failed image ingestion into the DLQ backend.

        Args:
            image_id: ID of the failed image ingestion
            metadata: Optional metadata to be stored alongside the image ID
        """

    @abstractmethod
    def get_queue_contents(self) -> list[str]:
        """Get the contents of the backend's Dead Letter Queue.

        Returns:
            List of image IDs for previously failed ingestions
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the DLQ backend."""


class StubbedDLQBackend(DLQBackend):
    """Stubbed DLQ Backend implementation.

    NOTE: This will be removed once the first DLQBackend (Redis) is implemented.
    """

    @override
    def insert(self, image_id: str, *, metadata: dict[str, object] | None = None):
        pass

    @override
    def get_queue_contents(self) -> list[str]:
        return []

    @property
    @override
    def name(self) -> str:
        return "stub-dlq-backend"

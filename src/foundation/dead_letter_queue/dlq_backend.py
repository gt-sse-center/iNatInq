"""DLQBackend interface - defines required behavior of objects serving as a dead letter queue backend."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing_extensions import override


class DLQBackend(ABC):
    """Interface defining expected behavior for a DLQ backend."""

    @abstractmethod
    def insert(self, image_id: str, *, metadata: dict[str, object] | None = None) -> None:
        """Insert ID of a failed image ingestion into the DLQ backend.

        Args:
            image_id: ID of the failed image ingestion
            metadata: Optional metadata to be stored alongside the image ID
        """

    @abstractmethod
    def get_queue_contents(self) -> Iterator[str]:
        """Get the contents of the backend's Dead Letter Queue.

        Returns:
            Iterator over previously failed image IDs

        Raises:
            UpstreamError: if backend fails to return contents
        """

    @abstractmethod
    def delete(self, image_ids: Iterable[str]) -> None:
        """Remove image IDs from the backend's Dead Letter Queue.

        Args:
            image_ids: iterable collection of image IDs to be deleted

        Raises:
            UpstreamError: if backend fails to delete IDs
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the DLQ backend."""


class StubbedDLQBackend(DLQBackend):
    """Stubbed DLQ Backend implementation.

    This implementation is used when the application has not configured a real backend.
    """

    @override
    def insert(self, image_id: str, *, metadata: dict[str, object] | None = None) -> None:
        pass

    @override
    def get_queue_contents(self) -> Iterator[str]:
        return iter([])

    @override
    def delete(self, image_ids: Iterable[str]) -> None:
        pass

    @property
    @override
    def name(self) -> str:
        return "stub-dlq-backend"

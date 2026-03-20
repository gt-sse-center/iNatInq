"""The `dead_letter_queue` module provides functionality for registering failed image ingestions to be retried at a later date."""

from foundation.dead_letter_queue.with_dlq import with_dlq
from foundation.dead_letter_queue.dlq import DLQ
from foundation.dead_letter_queue.dlq_backend_registry import get_dlq_backend

__all__ = [
    "DLQ",
    "get_dlq_backend",
    "with_dlq",
]

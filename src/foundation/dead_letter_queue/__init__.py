"""The `dead_letter_queue` module provides functionality for registering failed image ingestions to be retried at a later date."""

from foundation.dead_letter_queue.with_dlq import with_dlq
from foundation.dead_letter_queue.dlq import DLQ

__all__ = [
    "DLQ",
    "with_dlq",
]

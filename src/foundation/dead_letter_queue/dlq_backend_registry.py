"""Exposes a function for getting the global DLQBackend instance."""

import functools
import logging
import os

from foundation.dead_letter_queue.dlq_backend import DLQBackend
from foundation.dead_letter_queue.dlq_redis_backend import RedisDLQConfig, RedisDLQBackend

logger = logging.getLogger("foundation.dead_letter_queue")


def get_dlq_backend() -> DLQBackend | None:
    """Get the configured global DLQBackend instance, or None if not configured.

    Returns:
        Global DLQBackend instance or None if not configured

    Raises:
        ValueError: if 'DLQ_BACKEND' env variable is set to an invalid backend type.
        Other: if constructing the configured instance raises
    """

    # NOTE: '@functools.lru_cache' removes the docstring & alters the signature of the
    # wrapped function, so we've opted to wrap it in this public function instead
    return _get_dlq_backend()


_DLQ_BACKEND_CONSTRUCTORS = {
    "redis": lambda: RedisDLQBackend(RedisDLQConfig.from_env()),
}


@functools.lru_cache(maxsize=1)
def _get_dlq_backend() -> DLQBackend | None:
    backend_str = os.getenv("DLQ_BACKEND")
    if backend_str is None:
        logger.warning("DLQ_BACKEND env variable is not set. The dead letter queue will not be functional")
        return None

    backend = backend_str.lower()

    if backend not in _DLQ_BACKEND_CONSTRUCTORS:
        raise ValueError(
            f"Unsupported DLQ backend ({backend_str}), must be one of ({list(_DLQ_BACKEND_CONSTRUCTORS.keys())})"
        )
    return _DLQ_BACKEND_CONSTRUCTORS[backend]()

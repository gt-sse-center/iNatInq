"""In-memory semantic cache client. Follows the singleton pattern so only one cache instance is created."""

import threading

from qdrant_client import AsyncQdrantClient


class CacheClient:
    """In memory cache client. Follows the singleton pattern so only one cache instance is created."""

    _instance: "CacheClient | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Create a new cache client instance if it doesn't exist."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, timeout: int):
        with self._lock:
            if not hasattr(self, "initialized"):
                self.timeout = timeout
                self.client = AsyncQdrantClient(location=":memory:", timeout=timeout)
                self.initialized = True

    async def close(self) -> None:
        """Close the cache client."""
        await self.client.close()

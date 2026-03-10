"""Provides a DLQBackend Implementation using Redis."""

from collections.abc import Iterable, Iterator
import json
import logging
from typing import ClassVar, Self
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from typing_extensions import override
from foundation.dead_letter_queue.dlq_backend import DLQBackend
from foundation.exceptions import UpstreamError

logger = logging.getLogger("foundation.dead_letter_queue")


class RedisDLQConfig(BaseSettings):
    """Config object for Redis DLQ cache."""

    model_config: ClassVar = SettingsConfigDict(case_sensitive=False, env_file=None, populate_by_name=True)

    host: str = Field(validation_alias="DLQ_REDIS_HOST")
    port: int = Field(validation_alias="DLQ_REDIS_PORT")
    db_number: int = Field(validation_alias="DLQ_REDIS_DATABASE_NUMBER", ge=0)

    @classmethod
    def from_env(cls) -> Self:
        """Load instance of RedisConfig from environment variables."""
        # Pydantic automatically populates class fields with env variables
        return cls()  # pyright: ignore[reportCallIssue]


class RedisDLQBackend(DLQBackend):
    """Implementation of DLQBackend using Redis as the storage mechanism."""

    def __init__(self, config: RedisDLQConfig) -> None:
        self._client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db_number,
            decode_responses=True,
            retry=Retry(backoff=ExponentialBackoff(), retries=5),
        )

    @override
    def insert(self, image_id: str, *, metadata: dict[str, object] | None = None) -> None:
        try:
            metadata_str = json.dumps(metadata) if metadata is not None else ""
        except Exception:
            logger.warning(
                "Failed to serialize metadata dict",
                extra={
                    "image_id": image_id,
                    "metadata": metadata,
                },
            )
            metadata_str = ""
        # NOTE: we don't wrap failures in an UpstreamError, as the DLQ class (uses this backend) handles insertion errors
        self._client.set(image_id, metadata_str)

    @override
    def get_queue_contents(self) -> Iterator[str]:
        # NOTE: 'count' here is the size of the client's internal buffer, NOT the number of keys returned
        # This will iterate over all keys in the database
        try:
            for key in self._client.scan_iter(count=100):  # pyright: ignore[reportUnknownVariableType]
                assert isinstance(key, str), "Redis key should be a string"
                yield key
        except Exception as e:
            raise UpstreamError(f"Failed to retrieve image IDs from Redis DLQ backend: {e}") from e

    @override
    def delete(self, image_ids: Iterable[str]) -> None:
        keys = list(image_ids)
        if len(keys) == 0:
            return
        try:
            self._client.delete(*keys)
        except Exception as e:
            raise UpstreamError(f"Failed to delete image IDs from Redis DLQ backend: {e}") from e

    @property
    @override
    def name(self) -> str:
        return "redis-dlq-backend"

"""Integration tests for the Redis DLQ backend.

How to run:
    uv run pytest tests/integration/foundation/dlq/test_dlq_redis_backend.py
"""
# pyright: reportArgumentType=false

import json
import pytest
from collections.abc import Generator
from redis import Redis
from foundation.dead_letter_queue.dlq_redis_backend import RedisDLQConfig, RedisDLQBackend


pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def redis_client(redis_container: tuple[str, int]) -> Generator[Redis]:
    yield Redis(host=redis_container[0], port=redis_container[1], db=0, decode_responses=True)


@pytest.fixture(autouse=True)
def clear_cache(redis_client: Redis):
    """Empty Redis container cache between each test."""
    redis_client.flushdb()


@pytest.fixture
def redis_config(redis_container: tuple[str, int]) -> RedisDLQConfig:
    return RedisDLQConfig(host=redis_container[0], port=redis_container[1], db_number=0)


class TestRedisDLQBackend:
    """RedisDLQBackend Integration Tests"""

    def test_insert_adds_to_cache(self, redis_config: RedisDLQConfig, redis_client: Redis):
        """Redis backend insert() stores keys in actual Redis cache."""
        keys = {"1", "foo", "bar", "test", "42"}
        backend = RedisDLQBackend(redis_config)
        for key in keys:
            backend.insert(key)

        assert keys == set(redis_client.keys())

    def test_metadata_is_correctly_stored(self, redis_config: RedisDLQConfig, redis_client: Redis):
        """Redis backend stores metadata as JSON-serialized string in Redis."""
        key, metadata = "test", {"foo": 12345, "bar": "baz"}
        backend = RedisDLQBackend(redis_config)
        backend.insert(key, metadata=metadata)
        retrieved_metadata = redis_client.get(key)
        assert json.loads(retrieved_metadata) == metadata

    def test_get_queue_content_returns(self, redis_config: RedisDLQConfig, redis_client: Redis):
        """Redis backend get_queue_contents() retrieves all keys from actual Redis cache."""
        expected_keys = {"1", "2", "buckle", "my", "shoe"}
        for key in expected_keys:
            redis_client.set(key, "")

        backend = RedisDLQBackend(redis_config)
        retrieved_keys = set(backend.get_queue_contents())
        assert retrieved_keys == expected_keys

    def test_delete_removes_keys(self, redis_config: RedisDLQConfig, redis_client: Redis):
        """Redis backend delete() removes all specified keys from actual Redis cache."""
        keys = {"3", "4", "gimme", "some", "more"}
        for key in keys:
            redis_client.set(key, "")

        backend = RedisDLQBackend(redis_config)
        backend.delete(keys)
        assert len(redis_client.keys()) == 0

    def test_delete_empty_iterator_does_not_raise(self, redis_config: RedisDLQConfig):
        """Redis backend delete() handles empty key list without raising exceptions."""
        backend = RedisDLQBackend(redis_config)
        # Since the test will fail if this raises, no assert statement is needed
        backend.delete([])

    def test_smoke_test_large_cache_size(self, redis_config: RedisDLQConfig, redis_client: Redis):
        """Redis backend handles large key sets (2500+ keys) without performance degradation."""
        # This test isn't strictly necessary, but serves as a useful sanity check

        keys = [str(i) for i in range(2_500)]
        backend = RedisDLQBackend(redis_config)

        for key in keys:
            backend.insert(key)

        assert len(redis_client.keys()) == len(keys)

        returned_keys = list(backend.get_queue_contents())
        assert len(returned_keys) == len(keys)

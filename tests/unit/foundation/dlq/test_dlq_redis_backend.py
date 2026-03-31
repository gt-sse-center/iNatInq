"""Unit tests for the Redis DLQ backend.

How to run:
    uv run pytest tests/unit/foundation/dlq/test_dlq_redis_backend.py
"""

from collections.abc import Generator
import json
import pytest
from unittest.mock import ANY, MagicMock, patch
from foundation.dead_letter_queue.dlq_redis_backend import RedisDLQConfig, RedisDLQBackend
from foundation.exceptions import UpstreamError


@pytest.fixture
def mock_redis_client() -> Generator[MagicMock]:
    with patch("redis.Redis") as mock_redis_cls:
        mock_redis_instance = MagicMock()
        mock_redis_cls.return_value = mock_redis_instance
        yield mock_redis_instance


def create_redis_config() -> RedisDLQConfig:
    return RedisDLQConfig(host="localhost", port=1234, db_number=1)


class TestRedisConfig:
    def test_config_is_created_from_env_variables(self, monkeypatch: pytest.MonkeyPatch):
        """Redis config is created from DLQ_REDIS_* environment variables."""
        monkeypatch.setenv("DLQ_REDIS_HOST", "localhost")
        monkeypatch.setenv("DLQ_REDIS_PORT", "1234")
        monkeypatch.setenv("DLQ_REDIS_DATABASE_NUMBER", "1")
        config = RedisDLQConfig.from_env()
        assert config.host == "localhost"
        assert config.port == 1234
        assert config.db_number == 1


class TestRedisDLQBackend:
    class TestConstructor:
        def test_creates_redis_client_with_config_values(self):
            """Redis backend constructor creates Redis client with config values."""
            with patch("redis.Redis") as mock_redis_cls:
                config = create_redis_config()
                _ = RedisDLQBackend(config)
                mock_redis_cls.assert_called_once_with(
                    host=config.host,
                    port=config.port,
                    db=config.db_number,
                    decode_responses=True,
                    retry=ANY,
                )

    class TestNameProperty:
        def test_name_contains_redis(self):
            """Redis backend name contains 'redis' for identification."""
            backend = RedisDLQBackend(create_redis_config())
            assert "redis" in backend.name

    class TestInsert:
        def test_adds_to_cache(self, mock_redis_client: MagicMock):
            """Redis backend insert() calls Redis SET with the message ID."""
            backend = RedisDLQBackend(create_redis_config())
            backend.insert("id")
            mock_redis_client.set.assert_called_once_with("id", "")

        def test_converts_metadata_to_string(self, mock_redis_client: MagicMock):
            """Redis backend serializes metadata dict to JSON string before storage."""
            backend = RedisDLQBackend(create_redis_config())
            metadata: dict[str, object] = {"foo": 12345, "bar": "baz"}
            backend.insert("id", metadata=metadata)
            expected_val = json.dumps(metadata)
            mock_redis_client.set.assert_called_once_with("id", expected_val)

        def test_handles_failed_metadata_serialization(
            self, mock_redis_client: MagicMock, caplog: pytest.LogCaptureFixture
        ):
            """Redis backend falls back to empty string when metadata serialization fails."""
            backend = RedisDLQBackend(create_redis_config())
            with patch("json.dumps") as mock_json_dumps, caplog.at_level("WARNING"):
                mock_json_dumps.side_effect = Exception("oof")
                backend.insert("id", metadata={})
            mock_redis_client.set.assert_called_once_with("id", "")
            assert "Failed to serialize metadata" in caplog.text

    class TestGetQueueContents:
        def test_returns_cache_keys(self, mock_redis_client: MagicMock):
            """Redis backend get_queue_contents() returns all keys from scan_iter()."""
            cache_keys = ["1", "2", "3", "4", "5"]
            mock_redis_client.scan_iter.return_value = iter(cache_keys)
            backend = RedisDLQBackend(create_redis_config())

            key_iter = backend.get_queue_contents()
            assert list(key_iter) == cache_keys

        def test_raises_upstream_error_on_invalid_response(
            self,
            mock_redis_client: MagicMock,
        ):
            """Redis backend raises UpstreamError when scan_iter returns non-string keys."""
            mock_redis_client.scan_iter.return_value = iter([1, 2, 3, 4, 5])  # not strings
            backend = RedisDLQBackend(create_redis_config())

            with pytest.raises(UpstreamError, match="key should be a string"):
                # Drain iterator to trigger validation error
                list(backend.get_queue_contents())

        def test_raises_upstream_error_on_failure(self, mock_redis_client: MagicMock):
            """Redis backend raises UpstreamError when scan_iter fails with an exception."""
            mock_redis_client.scan_iter.side_effect = Exception("uh oh")
            backend = RedisDLQBackend(create_redis_config())

            with pytest.raises(UpstreamError, match="Failed to retrieve image IDs"):
                # Drain iterator to trigger error
                list(backend.get_queue_contents())

    class TestDelete:
        def test_delete_removes_keys(self, mock_redis_client: MagicMock):
            """Redis backend delete() calls Redis DELETE with all provided keys."""
            keys = {"1", "2", "3", "4", "5"}
            backend = RedisDLQBackend(create_redis_config())
            backend.delete(keys)

            mock_redis_client.delete.assert_called_once_with(*keys)

        def test_raises_upstream_error_on_failure(self, mock_redis_client: MagicMock):
            """Redis backend raises UpstreamError when delete operation fails with an exception."""
            mock_redis_client.delete.side_effect = Exception("oopsie")
            backend = RedisDLQBackend(create_redis_config())

            with pytest.raises(UpstreamError, match="Failed to delete image IDs"):
                backend.delete([""])

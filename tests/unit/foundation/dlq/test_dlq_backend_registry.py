"""Unit tests for the DLQ backend registry.

How to run:
    uv run pytest tests/unit/foundation/dlq/test_dlq_backend_registry.py
"""

import pytest
from unittest.mock import MagicMock
from foundation.dead_letter_queue import dlq_backend_registry
from foundation.dead_letter_queue.dlq_backend_registry import get_dlq_backend

BACKEND_TYPE_ENV_VARIABLE = "DLQ_BACKEND"


@pytest.fixture(autouse=True)
def clear_get_dlq_backend_cache():
    # Clear the lru_cache before each test
    dlq_backend_registry._get_dlq_backend.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_registry_returns_none_if_type_env_not_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.delenv(BACKEND_TYPE_ENV_VARIABLE, raising=False)
    with caplog.at_level("WARNING"):
        backend = get_dlq_backend()
    assert backend is None
    assert "DLQ_BACKEND env variable is not set" in caplog.text
    assert "dead letter queue will not be functional" in caplog.text


def test_registry_returns_redis_if_env_specifies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(BACKEND_TYPE_ENV_VARIABLE, "redis")
    mock_redis_cls = MagicMock()
    mock_redis_instance = MagicMock()
    mock_redis_cls.return_value = mock_redis_instance
    monkeypatch.setitem(dlq_backend_registry._DLQ_BACKEND_CONSTRUCTORS, "redis", mock_redis_cls)  # pyright: ignore[reportPrivateUsage]

    result = get_dlq_backend()
    assert result is mock_redis_instance


def test_registry_raises_if_invalid_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(BACKEND_TYPE_ENV_VARIABLE, "some-invalid-backend")
    with pytest.raises(ValueError, match=r"(?i)unsupported DLQ backend"):
        _ = get_dlq_backend()


def test_multiple_calls_return_same_instance():
    backend_1 = get_dlq_backend()
    backend_2 = get_dlq_backend()
    assert backend_1 is backend_2

"""Unit tests for the `@with_dlq` decorator.

How to run:
    uv run pytest tests/unit/foundation/dlq/test_with_dlq.py
"""

from unittest.mock import patch
import pytest
from foundation.dead_letter_queue.dlq import DLQ
from foundation.dead_letter_queue.dlq_backend import StubbedDLQBackend
from foundation.dead_letter_queue.with_dlq import with_dlq


class TestWithDLQ:
    def test_decorator_injects_dlq_instance(self):
        """@with_dlq decorator injects DLQ instance as first parameter to sync functions."""
        injected_dlq = None

        @with_dlq
        def test_fn(dlq: DLQ):
            nonlocal injected_dlq
            injected_dlq = dlq

        # Invoke function
        test_fn()

        assert isinstance(injected_dlq, DLQ)

    @pytest.mark.asyncio
    async def test_async_decorator_injects_dlq_instance(self):
        """@with_dlq decorator injects DLQ instance as first parameter to async functions."""
        injected_dlq = None

        @with_dlq
        async def test_fn(dlq: DLQ):
            nonlocal injected_dlq
            injected_dlq = dlq

        # Invoke function
        await test_fn()

        assert isinstance(injected_dlq, DLQ)

    def test_decorator_preserves_params(self):
        """@with_dlq decorator passes through all positional and keyword arguments unchanged."""
        captured_args = None
        captured_kwargs = None

        @with_dlq
        def test_fn(_dlq: DLQ, *args: object, **kwargs: object):
            nonlocal captured_args, captured_kwargs
            captured_args = args
            captured_kwargs = kwargs

        args = (1, "str", 1.2)
        kwargs = {"key": "value"}

        test_fn(*args, **kwargs)

        assert captured_args == args
        assert captured_kwargs == kwargs

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_params(self):
        """@with_dlq decorator passes through all arguments unchanged for async functions."""
        captured_args = None
        captured_kwargs = None

        @with_dlq
        async def test_fn(_dlq: DLQ, *args: object, **kwargs: object):
            nonlocal captured_args, captured_kwargs
            captured_args = args
            captured_kwargs = kwargs

        args = (1, "str", 1.2)
        kwargs = {"key": "value"}

        await test_fn(*args, **kwargs)

        assert captured_args == args
        assert captured_kwargs == kwargs

    def test_decorator_preserves_return_value(self):
        """@with_dlq decorator preserves function return values for sync functions."""

        @with_dlq
        def returns_int(_dlq: DLQ) -> int:
            return 0

        result = returns_int()
        assert result == 0

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_return_value(self):
        """@with_dlq decorator preserves function return values for async functions."""

        @with_dlq
        async def returns_int(_dlq: DLQ) -> int:
            return 0

        result = await returns_int()
        assert result == 0

    def test_decorator_adds_tag(self):
        """Ensure with_dlq adds '__wrapped_with_dlq' attribute"""

        @with_dlq
        def test_fn(_dlq: DLQ):
            pass

        assert hasattr(test_fn, "__wrapped_with_dlq")

    def test_decorator_uses_stub_if_no_backend_configured(self):
        """@with_dlq decorator falls back to stubbed backend when no real backend is configured."""
        with patch(
            "foundation.dead_letter_queue.dlq_backend_registry.get_dlq_backend"
        ) as mock_get_dlq_backend:
            mock_get_dlq_backend.return_value = None

            captured_backend = None

            @with_dlq
            def test_fn(dlq: DLQ):
                nonlocal captured_backend
                captured_backend = dlq._backend  # pyright: ignore[reportPrivateUsage]

            # Invoke function
            test_fn()

            assert isinstance(captured_backend, StubbedDLQBackend)

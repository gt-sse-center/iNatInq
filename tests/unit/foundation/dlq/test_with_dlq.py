import pytest
from foundation.dead_letter_queue.dlq import DLQ
from foundation.dead_letter_queue.with_dlq import with_dlq


class TestWithDLQ:
    def test_decorator_injects_dlq_instance(self):
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
        injected_dlq = None

        @with_dlq
        async def test_fn(dlq: DLQ):
            nonlocal injected_dlq
            injected_dlq = dlq

        # Invoke function
        await test_fn()

        assert isinstance(injected_dlq, DLQ)

    def test_decorator_preserves_params(self):
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
        @with_dlq
        def returns_int(_dlq: DLQ) -> int:
            return 0

        result = returns_int()
        assert result == 0

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_return_value(self):
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

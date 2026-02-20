"""Unit tests for foundation.async_utils module.

This module tests the async resource cleanup utilities used throughout the
application to safely close async resources.

# Test Coverage

- close_async_resource: Success path, error handling, custom close methods, edge cases
- run_coroutine: No-loop fallback, nested-loop support, return values, error propagation

# Running Tests

    uv run pytest tests/unit/foundation/test_async_utils.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundation.async_utils import close_async_resource, run_coroutine


class TestCloseAsyncResourceSuccess:
    """Tests for close_async_resource success paths."""

    @pytest.mark.asyncio
    async def test_awaits_close_and_logs_success(self):
        """Verify close method is awaited and success is logged.

        **Why this test is important:**
          - Core functionality: close must actually be called
          - Success logging is needed for debugging

        **What it tests:**
          - Resource close method is awaited
          - Debug log includes "closed successfully"
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "test_client")

        mock_resource.close.assert_awaited_once()
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert "closed successfully" in call_args[0][0]
        assert call_args[1]["extra"]["resource"] == "test_client"

    @pytest.mark.asyncio
    async def test_uses_custom_close_method(self):
        """Verify custom close method name is respected.

        **Why this test is important:**
          - Different libraries use different method names (close, disconnect, shutdown)
          - Users must be able to specify the correct cleanup method

        **What it tests:**
          - close_method parameter with "disconnect" value
          - Custom method is called instead of default "close"
        """
        mock_resource = MagicMock()
        mock_resource.disconnect = AsyncMock()

        await close_async_resource(mock_resource, "test_client", close_method="disconnect")

        mock_resource.disconnect.assert_awaited_once()


class TestCloseAsyncResourceErrors:
    """Tests for close_async_resource error handling."""

    @pytest.mark.asyncio
    async def test_logs_exception_on_close_failure(self):
        """Verify exception is logged when close fails.

        **Why this test is important:**
          - Close failures during shutdown must be logged, not silently ignored
          - Application should not crash during cleanup

        **What it tests:**
          - close_async_resource with failing close method
          - Error is caught and logged via logger.exception
          - Error message contains "Error closing"
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock(side_effect=ConnectionError("Connection lost"))

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "failing_resource")

        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "Error closing" in call_args[0][0]
        assert call_args[1]["extra"]["resource"] == "failing_resource"
        assert "Connection lost" in call_args[1]["extra"]["error"]

    @pytest.mark.asyncio
    async def test_does_not_raise_on_close_failure(self):
        """Verify close_async_resource never raises on failure.

        **Why this test is important:**
          - Cleanup code must not propagate exceptions
          - Prevents cascading failures during shutdown

        **What it tests:**
          - No exception is raised to caller despite close failure
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock(side_effect=RuntimeError("Boom"))

        with patch("foundation.async_utils.logger"):
            await close_async_resource(mock_resource, "failing_resource")
        # No exception raised


class TestCloseAsyncResourceEdgeCases:
    """Edge case tests for close_async_resource."""

    @pytest.mark.asyncio
    async def test_handles_resource_without_close_method(self):
        """Verify AttributeError is caught and logged for resource without close method.

        **Why this test is important:**
          - Resources without close method should not crash the application
          - The error is caught by the except Exception handler and logged

        **What it tests:**
          - Resource with no methods (spec=[])
          - AttributeError is caught (not propagated)
          - Error is logged via logger.exception
        """
        mock_resource = MagicMock(spec=[])  # No methods

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "no_close_resource")

        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "Error closing" in call_args[0][0]
        assert call_args[1]["extra"]["resource"] == "no_close_resource"

    @pytest.mark.asyncio
    async def test_handles_async_close_noop(self):
        """Verify resource with no-op async close method works.

        **Why this test is important:**
          - Some resources have async close that does nothing
          - These should still work without error

        **What it tests:**
          - Resource with async close that just passes
          - No error is raised, close completes successfully
        """
        mock_resource = MagicMock()

        async def async_close():
            pass

        mock_resource.close = async_close

        await close_async_resource(mock_resource, "noop_resource")

    @pytest.mark.asyncio
    async def test_multiple_resources_can_be_closed(self):
        """Verify multiple resources can be closed in sequence.

        **Why this test is important:**
          - Applications often need to close multiple resources during shutdown
          - Each close should be independent and complete

        **What it tests:**
          - Three resources closed in sequence
          - Each resource's close method is called exactly once
        """
        resources = [MagicMock() for _ in range(3)]
        for r in resources:
            r.close = AsyncMock()

        for i, resource in enumerate(resources):
            await close_async_resource(resource, f"resource_{i}")

        for resource in resources:
            resource.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_resource_name_is_allowed(self):
        """Verify empty resource name is allowed (though not recommended).

        **What it tests:**
          - Empty string passed as resource name
          - Close still executes successfully
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger"):
            await close_async_resource(mock_resource, "")

        mock_resource.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_special_characters_in_resource_name(self):
        """Verify resource names with special characters are handled.

        **What it tests:**
          - Resource name with slashes, colons, and at-sign
          - Close executes successfully
          - Name is correctly passed through to logging
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "resource/with:special@chars")

        mock_resource.close.assert_awaited_once()
        call_args = mock_logger.debug.call_args
        assert "resource/with:special@chars" in str(call_args)

    def test_called_via_asyncio_run(self):
        """Verify close_async_resource works when called via asyncio.run().

        **Why this test is important:**
          - The primary usage pattern is asyncio.run(close_async_resource(...))
          - Must work correctly in this scenario (e.g., Qdrant close())

        **What it tests:**
          - close_async_resource invoked via asyncio.run (from sync context)
          - Resource is properly closed
        """
        mock_resource = MagicMock()
        close_called = False

        async def mock_close():
            nonlocal close_called
            close_called = True

        mock_resource.close = mock_close

        with patch("foundation.async_utils.logger"):
            asyncio.run(close_async_resource(mock_resource, "run_resource"))

        assert close_called


class TestRunCoroutineNoLoop:
    """Tests for run_coroutine when no event loop is running."""

    def test_returns_coroutine_result(self):
        """Verify run_coroutine returns the value produced by the coroutine.

        **Why this test is important:**
          - Callers rely on the return value for downstream logic.

        **What it tests:**
          - A coroutine that returns 42 is executed via run_coroutine
          - The returned value is 42
        """

        async def compute():
            return 42

        assert run_coroutine(compute()) == 42

    def test_propagates_exception(self):
        """Verify exceptions raised inside the coroutine propagate to the caller.

        **Why this test is important:**
          - Errors must not be silently swallowed; callers need to handle them.

        **What it tests:**
          - A coroutine that raises ValueError
          - run_coroutine re-raises the same ValueError
        """

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_coroutine(fail())

    def test_executes_awaited_work(self):
        """Verify the coroutine body actually runs (side-effect check).

        **What it tests:**
          - A coroutine that mutates external state via a list append
          - The mutation is visible after run_coroutine returns
        """
        evidence: list[str] = []

        async def work():
            evidence.append("done")

        run_coroutine(work())
        assert evidence == ["done"]


class TestRunCoroutineInsideRunningLoop:
    """Tests for run_coroutine when an event loop is already running.

    This simulates the Databricks / Jupyter notebook scenario where
    ``asyncio.run()`` would fail with RuntimeError.
    """

    @pytest.mark.asyncio
    async def test_returns_result_inside_running_loop(self):
        """Verify run_coroutine succeeds when called from within a running loop.

        **Why this test is important:**
          - This is the exact scenario that triggers the Databricks bug.
          - run_coroutine must apply nest_asyncio and complete the coroutine.

        **What it tests:**
          - run_coroutine called while an asyncio loop is already running
          - The coroutine executes and its return value is propagated
        """

        async def compute():
            return "hello"

        result = run_coroutine(compute())
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_propagates_exception_inside_running_loop(self):
        """Verify exceptions propagate when called from a running loop.

        **What it tests:**
          - run_coroutine called from a running loop with a failing coroutine
          - The exception propagates to the caller
        """

        async def fail():
            raise RuntimeError("nested failure")

        with pytest.raises(RuntimeError, match="nested failure"):
            run_coroutine(fail())

    @pytest.mark.asyncio
    async def test_multiple_calls_inside_running_loop(self):
        """Verify run_coroutine can be called multiple times in the same loop.

        **Why this test is important:**
          - The Databricks entrypoints call run_coroutine for disable_indexing,
            then enable_indexing, then close() — all in the same loop.

        **What it tests:**
          - Three sequential run_coroutine calls inside one running loop
          - Each call completes independently and returns its result
        """
        results = []
        for i in range(3):

            async def compute(val=i):
                return val * 10

            results.append(run_coroutine(compute()))

        assert results == [0, 10, 20]

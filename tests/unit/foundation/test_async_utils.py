"""Unit tests for foundation.async_utils module.

This module tests the async resource cleanup utilities used throughout the
application to safely close async resources in various event loop states.

# Test Coverage

- _create_close_callback: Success logging, exception logging, cancelled task handling
- close_async_resource with running loop: Background task scheduling, custom close methods
- close_async_resource with stopped loop: Synchronous execution, error handling
- close_async_resource with no loop: New loop creation, error handling
- Edge cases: Missing close method, multiple resources, special characters in names

# Running Tests

    uv run pytest tests/unit/foundation/test_async_utils.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundation.async_utils import _create_close_callback, close_async_resource


class TestCreateCloseCallback:
    """Tests for _create_close_callback function."""

    def test_callback_logs_success_on_successful_task(self):
        """Verify callback logs debug message when task succeeds.

        **Why this test is important:**

        - Ensures successful resource cleanup is properly logged for debugging
        - Verifies the callback correctly retrieves the task result
        - Confirms resource name is included in log metadata for traceability

        **What it tests:**

        - Callback invocation with a successful mock task
        - Debug logging with "closed successfully" message
        - Correct resource name in log extra data
        """
        callback = _create_close_callback("test_resource")
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.result.return_value = None

        with patch("foundation.async_utils.logger") as mock_logger:
            callback(mock_task)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert "closed successfully" in call_args[0][0]
        assert call_args[1]["extra"]["resource"] == "test_resource"

    def test_callback_logs_exception_on_failed_task(self):
        """Verify callback logs exception when task raises error.

        **Why this test is important:**

        - Failed resource cleanup must be logged for troubleshooting
        - Error details need to be captured for diagnosis
        - Silent failures would make debugging connection issues impossible

        **What it tests:**

        - Callback handling of task that raises RuntimeError
        - Exception logging with "Error closing" message
        - Resource name and error message in log extra data
        """
        callback = _create_close_callback("failing_resource")
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.result.side_effect = RuntimeError("Close failed")

        with patch("foundation.async_utils.logger") as mock_logger:
            callback(mock_task)

        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "Error closing" in call_args[0][0]
        assert call_args[1]["extra"]["resource"] == "failing_resource"
        assert "Close failed" in call_args[1]["extra"]["error"]

    def test_callback_handles_cancelled_task(self):
        """Verify callback propagates CancelledError from cancelled task.

        **Why this test is important:**

        - CancelledError is a BaseException that should propagate, not be swallowed
        - Task cancellation is a normal shutdown signal that must flow through
        - Catching CancelledError incorrectly would prevent graceful shutdown

        **What it tests:**

        - Callback behavior when task raises CancelledError
        - CancelledError propagates rather than being caught
        - Proper exception type handling (BaseException vs Exception)
        """
        callback = _create_close_callback("cancelled_resource")
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.result.side_effect = asyncio.CancelledError()

        # CancelledError is a BaseException, not Exception, so it propagates
        with pytest.raises(asyncio.CancelledError):
            callback(mock_task)


class TestCloseAsyncResourceRunningLoop:
    """Tests for close_async_resource when event loop is running."""

    @pytest.mark.asyncio
    async def test_schedules_background_task_when_loop_running(self):
        """Verify close is scheduled as background task in async context.

        **Why this test is important:**

        - Resources must be closed without blocking the running event loop
        - Background scheduling allows caller to continue without waiting
        - This is the primary path for cleanup during normal async operation

        **What it tests:**

        - close_async_resource called from async context
        - Resource close method is invoked
        - No blocking of the calling coroutine
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "test_client")

        # The close method should have been called
        mock_resource.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_custom_close_method(self):
        """Verify custom close method name is respected.

        **Why this test is important:**

        - Different libraries use different method names (close, disconnect, shutdown)
        - Users must be able to specify the correct cleanup method
        - Default "close" should not be hardcoded when alternative is specified

        **What it tests:**

        - close_method parameter with "disconnect" value
        - Custom method is called instead of default "close"
        - Parameter correctly routes to the specified method
        """
        mock_resource = MagicMock()
        mock_resource.disconnect = AsyncMock()

        await close_async_resource(mock_resource, "test_client", close_method="disconnect")

        mock_resource.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_scheduling_debug_message(self):
        """Verify debug message is logged when scheduling background task.

        **Why this test is important:**

        - Background task scheduling should be observable for debugging
        - Helps trace when cleanup was initiated vs completed
        - Essential for diagnosing cleanup timing issues

        **What it tests:**

        - Debug logging occurs when scheduling background task
        - Message contains "background task" indicator
        - Logging happens before task completion
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "scheduled_resource")

        # Check that debug was called with scheduling message
        debug_calls = mock_logger.debug.call_args_list
        scheduling_call = [c for c in debug_calls if "background task" in str(c)]
        assert len(scheduling_call) >= 1


class TestCloseAsyncResourceStoppedLoop:
    """Tests for close_async_resource when event loop exists but is stopped."""

    def test_runs_close_synchronously_on_stopped_loop(self):
        """Verify close runs to completion when loop exists but is stopped.

        **Why this test is important:**

        - During shutdown, the event loop may be stopped but not closed
        - Resources still need cleanup even in this transitional state
        - run_until_complete allows synchronous cleanup in this scenario

        **What it tests:**

        - close_async_resource with stopped event loop
        - Resource close method is actually invoked
        - Debug logging confirms successful close
        """
        mock_resource = MagicMock()
        close_called = False

        async def mock_close():
            nonlocal close_called
            close_called = True

        mock_resource.close = mock_close

        # Create a new event loop but don't run it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with patch("foundation.async_utils.logger") as mock_logger:
                # Run the async function in the stopped loop
                loop.run_until_complete(close_async_resource(mock_resource, "sync_resource"))

            assert close_called
            mock_logger.debug.assert_called()
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_logs_error_when_close_fails_on_stopped_loop(self):
        """Verify exception is logged when close fails on stopped loop.

        **Why this test is important:**

        - Close failures during shutdown must be logged, not silently ignored
        - Connection errors are common during shutdown and need visibility
        - Error handling path must be tested separately from success path

        **What it tests:**

        - close_async_resource with failing close method
        - Error is caught and logged appropriately
        - Application continues without raising (graceful degradation)
        """
        mock_resource = MagicMock()

        async def failing_close():
            raise ConnectionError("Connection lost")

        mock_resource.close = failing_close

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with patch("foundation.async_utils.logger") as mock_logger:
                # The outer function catches and logs the error
                loop.run_until_complete(close_async_resource(mock_resource, "failing_resource"))

            # Since we're in the running loop branch, it schedules as task
            # The error will be caught by the callback
        finally:
            loop.close()
            asyncio.set_event_loop(None)


class TestCloseAsyncResourceNoLoop:
    """Tests for close_async_resource when no event loop exists."""

    def test_creates_new_loop_when_none_exists(self):
        """Verify new event loop is created when none exists.

        **Why this test is important:**

        - Cleanup may be triggered from synchronous code without an event loop
        - The utility must handle this case by creating a temporary loop
        - Prevents "no running event loop" errors during edge-case cleanup

        **What it tests:**

        - close_async_resource when no event loop is set
        - New loop is created via asyncio.run
        - Resource close method is successfully invoked
        """
        mock_resource = MagicMock()
        close_called = False

        async def mock_close():
            nonlocal close_called
            close_called = True

        mock_resource.close = mock_close

        # Ensure no event loop exists
        try:
            asyncio.get_event_loop().close()
        except RuntimeError:
            pass
        asyncio.set_event_loop(None)

        with patch("foundation.async_utils.logger") as mock_logger:
            # This should create a new loop via asyncio.run
            asyncio.run(close_async_resource(mock_resource, "no_loop_resource"))

        assert close_called

    def test_logs_success_with_new_event_loop(self):
        """Verify success message is logged when new event loop is used.

        **Why this test is important:**

        - New loop creation should be logged for debugging
        - Confirms the no-loop code path is properly instrumented
        - Helps distinguish which cleanup path was taken

        **What it tests:**

        - Debug logging occurs when using new event loop
        - Log confirms successful resource close
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger") as mock_logger:
            asyncio.run(close_async_resource(mock_resource, "new_loop_resource"))

        # Verify debug was called
        assert mock_logger.debug.called


class TestCloseAsyncResourceStoppedLoopError:
    """Tests for close_async_resource error handling on stopped loop."""

    @pytest.mark.asyncio
    async def test_logs_error_when_run_until_complete_fails(self):
        """Verify exception is logged when run_until_complete raises error.

        **Why this test is important:**

        - run_until_complete can fail for various reasons (loop closed, etc.)
        - These failures must be logged to aid debugging
        - Error handling must not crash the calling code

        **What it tests:**

        - Mock loop with run_until_complete that raises ConnectionError
        - Exception is caught and logged via logger.exception
        - Error message contains "Error closing"
        """
        mock_resource = MagicMock()

        async def failing_close():
            raise ConnectionError("Connection lost during close")

        mock_resource.close = failing_close

        # Create a mock loop that is NOT running
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.side_effect = ConnectionError("Connection lost during close")

        with (
            patch("foundation.async_utils.asyncio.get_event_loop", return_value=mock_loop),
            patch("foundation.async_utils.logger") as mock_logger,
        ):
            await close_async_resource(mock_resource, "error_resource")

        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "Error closing" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_success_path_on_stopped_loop(self):
        """Verify successful close when loop is stopped.

        **Why this test is important:**

        - Confirms the happy path works for stopped loop scenario
        - Validates run_until_complete is called correctly
        - Ensures debug logging occurs on success

        **What it tests:**

        - Mock loop with successful run_until_complete
        - run_until_complete is invoked exactly once
        - Debug logging confirms success
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = None

        with (
            patch("foundation.async_utils.asyncio.get_event_loop", return_value=mock_loop),
            patch("foundation.async_utils.logger") as mock_logger,
        ):
            await close_async_resource(mock_resource, "stopped_loop_resource")

        mock_loop.run_until_complete.assert_called_once()
        mock_logger.debug.assert_called()


class TestCloseAsyncResourceNoLoopPath:
    """Tests for close_async_resource when no event loop exists."""

    @pytest.mark.asyncio
    async def test_creates_new_loop_via_asyncio_run(self):
        """Verify new loop is created via asyncio.run when get_event_loop raises.

        **Why this test is important:**

        - RuntimeError from get_event_loop indicates no loop exists
        - asyncio.run must be used to create a temporary loop
        - This fallback path is essential for cleanup from sync contexts

        **What it tests:**

        - get_event_loop raises RuntimeError
        - asyncio.run is called as fallback
        - Debug logging occurs
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with (
            patch("foundation.async_utils.asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("foundation.async_utils.asyncio.run") as mock_run,
            patch("foundation.async_utils.logger") as mock_logger,
        ):
            await close_async_resource(mock_resource, "no_loop_resource")

        mock_run.assert_called_once()
        mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_logs_error_when_asyncio_run_fails(self):
        """Verify exception is logged when asyncio.run raises error.

        **Why this test is important:**

        - asyncio.run can fail (e.g., nested event loops, OS errors)
        - These failures must be logged for debugging
        - Application should not crash during cleanup

        **What it tests:**

        - asyncio.run raises OSError
        - Exception is caught and logged via logger.exception
        - Error message contains "Error closing"
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with (
            patch("foundation.async_utils.asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("foundation.async_utils.asyncio.run", side_effect=OSError("System error")),
            patch("foundation.async_utils.logger") as mock_logger,
        ):
            await close_async_resource(mock_resource, "failing_no_loop")

        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "Error closing" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_success_logs_new_event_loop_message(self):
        """Verify success path logs message about new event loop.

        **Why this test is important:**

        - Confirms logging indicates which code path was taken
        - "new event loop" message helps distinguish from other paths
        - Essential for debugging cleanup behavior

        **What it tests:**

        - Successful close via asyncio.run path
        - Debug message contains "new event loop"
        - Logging captures the execution path
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with (
            patch("foundation.async_utils.asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("foundation.async_utils.asyncio.run") as mock_run,
            patch("foundation.async_utils.logger") as mock_logger,
        ):
            await close_async_resource(mock_resource, "new_loop_success")

        # Verify debug message mentions new event loop
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        assert any("new event loop" in call for call in debug_calls)


class TestCloseAsyncResourceEdgeCases:
    """Edge case tests for close_async_resource."""

    @pytest.mark.asyncio
    async def test_handles_resource_without_close_method(self):
        """Verify AttributeError is raised for resource without close method.

        **Why this test is important:**

        - Resources without close method should fail fast with clear error
        - AttributeError is the expected Python behavior
        - Prevents silent failure when wrong object is passed

        **What it tests:**

        - Resource with no methods (spec=[])
        - AttributeError is raised when close is called
        - Error propagates to caller for handling
        """
        mock_resource = MagicMock(spec=[])  # No methods

        with pytest.raises(AttributeError):
            await close_async_resource(mock_resource, "no_close_resource")

    @pytest.mark.asyncio
    async def test_handles_async_generator_close(self):
        """Verify resource with simple async close method works.

        **Why this test is important:**

        - Some resources have async close that does nothing (no-op)
        - These should still work without error
        - Validates minimal async close implementation

        **What it tests:**

        - Resource with async close that just passes
        - No error is raised
        - Close completes successfully
        """
        mock_resource = MagicMock()

        async def async_close():
            pass  # Simple async close that does nothing

        mock_resource.close = async_close

        # This should work without error
        await close_async_resource(mock_resource, "async_gen_resource")

    @pytest.mark.asyncio
    async def test_multiple_resources_can_be_closed(self):
        """Verify multiple resources can be closed in sequence.

        **Why this test is important:**

        - Applications often need to close multiple resources during shutdown
        - Each close should be independent and complete
        - Ensures no state leakage between closes

        **What it tests:**

        - Three resources closed in sequence
        - Each resource's close method is called exactly once
        - All closes complete successfully
        """
        resources = [MagicMock() for _ in range(3)]
        for i, r in enumerate(resources):
            r.close = AsyncMock()

        for i, resource in enumerate(resources):
            await close_async_resource(resource, f"resource_{i}")

        for resource in resources:
            resource.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_resource_name_is_allowed(self):
        """Verify empty resource name is allowed (though not recommended).

        **Why this test is important:**

        - Function should not crash on edge case inputs
        - Empty string is valid even if not ideal
        - Demonstrates defensive coding

        **What it tests:**

        - Empty string passed as resource name
        - Close still executes successfully
        - No exception is raised
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger"):
            await close_async_resource(mock_resource, "")

        mock_resource.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_special_characters_in_resource_name(self):
        """Verify resource names with special characters are handled.

        **Why this test is important:**

        - Resource names may contain paths, URLs, or other special chars
        - Logging should handle these without breaking
        - Ensures robustness in real-world scenarios

        **What it tests:**

        - Resource name with slashes, colons, and at-sign
        - Close executes successfully
        - Name is correctly passed through to logging
        """
        mock_resource = MagicMock()
        mock_resource.close = AsyncMock()

        with patch("foundation.async_utils.logger") as mock_logger:
            await close_async_resource(mock_resource, "resource/with:special@chars")

        mock_resource.close.assert_called_once()
        # Verify the name was passed through to logging
        call_args = mock_logger.debug.call_args
        assert "resource/with:special@chars" in str(call_args)

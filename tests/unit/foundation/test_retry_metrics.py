"""Unit tests for retry Prometheus metrics.

Verifies that retry attempt counts and exhaustion rates are emitted
correctly from create_retry_logger, async_retry_call, and RetryWithBackoff.
"""

import logging
import pytest
from prometheus_client import REGISTRY

from unittest.mock import MagicMock
from foundation.retry import RetryWithBackoff, async_retry_call, create_retry_logger
from foundation.exceptions import UpstreamError


def _attempts(client: str, operation: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "inatinq_retry_attempts_total",
            {"client": client, "operation": operation, "outcome": outcome},
        )
        or 0.0
    )


def _exhaustions(client: str, operation: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "inatinq_retry_exhaustions_total",
            {"client": client, "operation": operation},
        )
        or 0.0
    )


class TestCreateRetryLoggerMetrics:
    """Prometheus metric assertions for create_retry_logger."""

    def test_retry_attempt_increments_counter(self) -> None:
        """Each call to the logger callback increments retry attempts with outcome=retry.

        Calling log_retry twice increments the counter by 2.
        """
        logger = logging.getLogger("test.retry_logger")
        log_retry = create_retry_logger(logger, client="test_client", operation="test_op")

        retry_state = MagicMock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = ConnectionError("boom")
        retry_state.next_action.sleep = 1.0
        retry_state.attempt_number = 1

        before = _attempts("test_client", "test_op", "retry")
        log_retry(retry_state)
        log_retry(retry_state)
        assert _attempts("test_client", "test_op", "retry") == before + 2.0

    def test_no_metric_when_outcome_not_failed(self) -> None:
        """Logger does not emit when the outcome did not fail.

        Calling log_retry with a non-failed outcome emits nothing.
        """
        logger = logging.getLogger("test.retry_logger_nofail")
        log_retry = create_retry_logger(logger, client="test_nofail", operation="op")

        retry_state = MagicMock()
        retry_state.outcome.failed = False

        before = _attempts("test_nofail", "op", "retry")
        log_retry(retry_state)
        assert _attempts("test_nofail", "op", "retry") == before

    def test_no_metric_when_client_not_set(self) -> None:
        """No metrics are emitted when client is not provided (opt-in behaviour).

        Calling log_retry without client produces no counter sample.
        """
        logger = logging.getLogger("test.retry_no_client")
        log_retry = create_retry_logger(logger)

        retry_state = MagicMock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = ConnectionError("boom")
        retry_state.next_action.sleep = 1.0
        retry_state.attempt_number = 1

        # Should not raise and should not register a sample for client=""
        log_retry(retry_state)
        assert (
            REGISTRY.get_sample_value(
                "inatinq_retry_attempts_total", {"client": "", "operation": "", "outcome": "retry"}
            )
            is None
        )


class TestAsyncRetryCallMetrics:
    """Prometheus metric assertions for async_retry_call."""

    @pytest.mark.asyncio
    async def test_retry_attempt_increments_counter(self) -> None:
        """Each failed retriable attempt increments retry attempts with outcome=retry.

        Two retriable failures before success → counter increments by 2.
        """
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        before = _attempts("async_client", "async_op", "retry")
        await async_retry_call(
            flaky,
            max_retries=5,
            min_wait=0,
            max_wait=0,
            is_retriable=lambda e: isinstance(e, ConnectionError),
            operation="async_op",
            client="async_client",
        )
        assert _attempts("async_client", "async_op", "retry") == before + 2.0

    @pytest.mark.asyncio
    async def test_success_after_retry_recorded(self) -> None:
        """A successful call after one or more retries records outcome=success_after_retry.

        One retry then success → success_after_retry increments by 1, retry by 1.
        """
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient")
            return "ok"

        before_sar = _attempts("sar_client", "sar_op", "success_after_retry")
        before_retry = _attempts("sar_client", "sar_op", "retry")
        result = await async_retry_call(
            flaky,
            max_retries=3,
            min_wait=0,
            max_wait=0,
            is_retriable=lambda e: isinstance(e, ConnectionError),
            operation="sar_op",
            client="sar_client",
        )
        assert result == "ok"
        assert _attempts("sar_client", "sar_op", "success_after_retry") == before_sar + 1.0
        assert _attempts("sar_client", "sar_op", "retry") == before_retry + 1.0

    @pytest.mark.asyncio
    async def test_exhaustion_increments_counter(self) -> None:
        """All retries exhausted increments both retry_attempts_total and retry_exhaustions_total.

        max_retries=2, always fails → exhausted=1, exhaustions=1.
        """

        async def always_fails() -> None:
            raise ConnectionError("permanent")

        before_ex = _attempts("ex_client", "ex_op", "exhausted")
        before_total = _exhaustions("ex_client", "ex_op")

        with pytest.raises(UpstreamError):
            await async_retry_call(
                always_fails,
                max_retries=2,
                min_wait=0,
                max_wait=0,
                is_retriable=lambda e: isinstance(e, ConnectionError),
                operation="ex_op",
                client="ex_client",
            )

        assert _attempts("ex_client", "ex_op", "exhausted") == before_ex + 1.0
        assert _exhaustions("ex_client", "ex_op") == before_total + 1.0

    @pytest.mark.asyncio
    async def test_no_retry_no_metrics(self) -> None:
        """A successful first attempt emits no retry metrics.

        First attempt succeeds → no retry, success_after_retry, or exhausted.
        """

        async def succeeds() -> str:
            return "ok"

        before_retry = _attempts("clean_client", "clean_op", "retry")
        before_sar = _attempts("clean_client", "clean_op", "success_after_retry")
        before_ex = _attempts("clean_client", "clean_op", "exhausted")

        await async_retry_call(
            succeeds,
            max_retries=3,
            min_wait=0,
            max_wait=0,
            is_retriable=lambda e: True,
            operation="clean_op",
            client="clean_client",
        )

        assert _attempts("clean_client", "clean_op", "retry") == before_retry
        assert _attempts("clean_client", "clean_op", "success_after_retry") == before_sar
        assert _attempts("clean_client", "clean_op", "exhausted") == before_ex

    @pytest.mark.asyncio
    async def test_labels_propagated_from_caller(self) -> None:
        """client and operation labels on emitted metrics match the values passed by the caller.

        client="myservice", operation="myop" → those exact labels appear in the counter.
        """

        async def always_fails() -> None:
            raise ConnectionError("boom")

        with pytest.raises(UpstreamError):
            await async_retry_call(
                always_fails,
                max_retries=1,
                min_wait=0,
                max_wait=0,
                is_retriable=lambda e: isinstance(e, ConnectionError),
                operation="myop",
                client="myservice",
            )

        assert _exhaustions("myservice", "myop") >= 1.0


class TestRetryWithBackoffMetrics:
    """Prometheus metric assertions for RetryWithBackoff."""

    def test_retry_attempt_increments_counter(self) -> None:
        """Each retry in RetryWithBackoff increments retry_attempts_total with outcome=retry.

        Two retriable failures before success → counter increments by 2.
        """
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "done"

        retry = RetryWithBackoff(
            max_attempts=5,
            wait_min=0,
            wait_max=0,
            retry_exceptions=(ValueError,),
            client="rwb_client",
            operation="rwb_op",
        )

        before = _attempts("rwb_client", "rwb_op", "retry")
        retry.call(flaky)
        assert _attempts("rwb_client", "rwb_op", "retry") == before + 2.0

    def test_success_after_retry_recorded(self) -> None:
        """A successful call after one or more retries records outcome=success_after_retry.

        One retry then success → success_after_retry increments by 1, retry by 1.
        """
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient")
            return "done"

        retry = RetryWithBackoff(
            max_attempts=5,
            wait_min=0,
            wait_max=0,
            retry_exceptions=(ValueError,),
            client="client",
            operation="op",
        )

        before_sar = _attempts("client", "op", "success_after_retry")
        before_retry = _attempts("client", "op", "retry")
        result = retry.call(flaky)
        assert result == "done"
        after_sar = _attempts("client", "op", "success_after_retry")
        after_retry = _attempts("client", "op", "retry")
        assert after_sar == before_sar + 1.0
        assert after_retry == before_retry + 1.0

    def test_exhaustion_increments_counter(self) -> None:
        """All retries exhausted in RetryWithBackoff increments both exhaustion counters.

        max_attempts=2, always fails → exhausted=1, exhaustions=1.
        """

        def always_fails() -> None:
            raise ValueError("permanent")

        retry = RetryWithBackoff(
            max_attempts=2,
            wait_min=0,
            wait_max=0,
            retry_exceptions=(ValueError,),
            client="rwb_ex_client",
            operation="rwb_ex_op",
        )

        before_ex = _attempts("rwb_ex_client", "rwb_ex_op", "exhausted")
        before_total = _exhaustions("rwb_ex_client", "rwb_ex_op")

        with pytest.raises(ValueError):
            retry.call(always_fails)

        assert _attempts("rwb_ex_client", "rwb_ex_op", "exhausted") == before_ex + 1.0
        assert _exhaustions("rwb_ex_client", "rwb_ex_op") == before_total + 1.0

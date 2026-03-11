"""Unit tests for circuit breaker Prometheus metrics (Story 1.3).

Verifies that CircuitBreakerListener and AsyncCircuitBreakerListener
emit the correct Prometheus metrics on state transitions.

# Test Coverage

The tests cover:
  - Gauge updates on state transitions (closed, open, half_open)
  - Transition counter increments on each state change
  - half_open maps to gauge value 2
  - Recovery path (open → half_open → closed) is recorded correctly
  - Initial gauge is set to 0 (closed) on breaker creation

# Running Tests

Run with: uv run pytest tests/unit/foundation/test_circuit_breaker_metrics.py -v
"""

from typing import cast
from unittest.mock import MagicMock

import aiobreaker
import pybreaker
import pytest
from prometheus_client import REGISTRY

from foundation.circuit_breaker import (
    AsyncCircuitBreakerListener,
    CircuitBreakerListener,
    create_async_circuit_breaker,
    create_circuit_breaker,
)


def _make_aio_state(
    enum_value: aiobreaker.state.CircuitBreakerState,
) -> aiobreaker.state.CircuitBreakerBaseState:
    """Create a mock aiobreaker state instance whose state is the given enum value."""
    mock = MagicMock(spec=aiobreaker.state.CircuitBreakerBaseState)
    mock.state = enum_value
    return cast(aiobreaker.state.CircuitBreakerBaseState, mock)


def _make_pybreaker_state(name: str) -> pybreaker.CircuitBreakerState:
    """Create a mock pybreaker state instance with the given state name."""
    mock = MagicMock(spec=pybreaker.CircuitBreakerState)
    mock.name = name
    return cast(pybreaker.CircuitBreakerState, mock)


def _gauge(breaker_name: str) -> float | None:
    """Return the current Prometheus gauge value for the given breaker."""
    return REGISTRY.get_sample_value(
        "inatinq_circuit_breaker_state",
        {"breaker": breaker_name},
    )


def _transitions(
    breaker_name: str, from_state: pybreaker.CircuitBreakerState, to_state: pybreaker.CircuitBreakerState
) -> float:
    """Return the current Prometheus transition count value for the given breaker from a given state to another given state."""
    state1 = from_state.name
    state2 = to_state.name
    return (
        REGISTRY.get_sample_value(
            "inatinq_circuit_breaker_transitions_total",
            {"breaker": breaker_name, "from_state": state1, "to_state": state2},
        )
        or 0.0
    )


class TestCircuitBreakerListenerMetrics:
    """Prometheus metric assertions for the sync CircuitBreakerListener."""

    closed = _make_pybreaker_state("closed")
    open = _make_pybreaker_state("open")
    half_open = _make_pybreaker_state("half_open")

    def test_initial_state_is_closed(self) -> None:
        """Gauge starts at 0 (closed) when a breaker is created."""
        name = "sync_initial_state_test"
        cb = create_circuit_breaker(name=name)
        assert cb.state.name == "closed"
        assert _gauge(name) == 0

    def test_state_gauge_updates_on_transition(self) -> None:
        """Ensures the gauge reflects new state after all transitions.
        Also ensures the gauge values are mapped correctly."""
        name = "sync_gauge_test"
        listener = CircuitBreakerListener()
        cb = create_circuit_breaker(name=name)
        assert _gauge(name) == 0
        listener.state_change(cb, self.closed, self.open)
        assert _gauge(name) == 1
        listener.state_change(cb, self.open, self.closed)
        assert _gauge(name) == 0
        listener.state_change(cb, self.closed, self.half_open)
        assert _gauge(name) == 2
        listener.state_change(cb, self.half_open, self.open)
        assert _gauge(name) == 1
        listener.state_change(cb, self.open, self.half_open)
        assert _gauge(name) == 2
        listener.state_change(cb, self.half_open, self.closed)
        assert _gauge(name) == 0

    def test_transition_counter_increments(self) -> None:
        """Ensures the counter increments on each unique state change."""
        name = "sync_counter_test"
        listener = CircuitBreakerListener()
        cb = create_circuit_breaker(name)
        listener.state_change(cb, self.closed, self.open)
        listener.state_change(cb, self.open, self.closed)
        closed_to_open = _transitions(name, self.closed, self.open)
        open_to_closed = _transitions(name, self.open, self.closed)
        assert closed_to_open == 1
        assert open_to_closed == 1
        listener.state_change(cb, self.closed, self.open)
        second_closed_to_open = _transitions(name, self.closed, self.open)
        no_second_open_to_closed = _transitions(name, self.open, self.closed)
        assert second_closed_to_open == 2
        assert no_second_open_to_closed == 1

    def test_unknown_state_raises_key_error(self) -> None:
        """state_change raises KeyError when new_state is not a recognised state name."""
        listener = CircuitBreakerListener()
        cb = create_circuit_breaker("sync_unknown_state_test")
        with pytest.raises(KeyError):
            listener.state_change(cb, self.closed, _make_pybreaker_state("unknown"))

    def test_recovery_transition_recorded(self) -> None:
        """Open → half_open → closed recovery path is tracked correctly."""
        name = "sync_recovery_test"
        listener = CircuitBreakerListener()
        cb = create_circuit_breaker(name)
        before_open_to_half = _transitions(name, self.open, self.half_open)
        before_half_to_closed = _transitions(name, self.half_open, self.closed)
        listener.state_change(cb, self.open, self.half_open)
        listener.state_change(cb, self.half_open, self.closed)
        after_open_to_half = _transitions(name, self.open, self.half_open)
        after_half_to_closed = _transitions(name, self.half_open, self.closed)
        assert after_open_to_half == before_open_to_half + 1
        assert after_half_to_closed == before_half_to_closed + 1
        assert _gauge(name) == 0


class TestAsyncCircuitBreakerListenerMetrics:
    """Prometheus metric assertions for the AsyncCircuitBreakerListener."""

    closed = _make_aio_state(aiobreaker.state.CircuitBreakerState.CLOSED)
    open = _make_aio_state(aiobreaker.state.CircuitBreakerState.OPEN)
    half_open = _make_aio_state(aiobreaker.state.CircuitBreakerState.HALF_OPEN)
    py_closed = _make_pybreaker_state("closed")
    py_open = _make_pybreaker_state("open")
    py_half_open = _make_pybreaker_state("half_open")

    def test_initial_state_is_closed(self) -> None:
        """Gauge starts at 0 (closed) when an async breaker is created."""
        name = "async_initial_state_test"
        cb = create_async_circuit_breaker(name=name)
        assert cb.state.state.name.lower() == "closed"
        assert _gauge(name) == 0.0

    def test_state_gauge_updates_on_transition(self) -> None:
        """Ensures the gauge reflects new state after all transitions.
        Also ensures the gauge values are mapped correctly."""
        name = "async_gauge_test"
        listener = AsyncCircuitBreakerListener()
        cb = create_async_circuit_breaker(name)
        listener.state_change(cb, self.closed, self.open)
        assert _gauge(name) == 1
        listener.state_change(cb, self.open, self.closed)
        assert _gauge(name) == 0
        listener.state_change(cb, self.closed, self.half_open)
        assert _gauge(name) == 2
        listener.state_change(cb, self.half_open, self.open)
        assert _gauge(name) == 1
        listener.state_change(cb, self.closed, self.half_open)
        assert _gauge(name) == 2
        listener.state_change(cb, self.half_open, self.closed)
        assert _gauge(name) == 0

    def test_transition_counter_increments(self) -> None:
        """Ensures the counter increments on each unique state change."""
        name = "async_counter_test"
        listener = AsyncCircuitBreakerListener()
        cb = create_async_circuit_breaker(name)
        listener.state_change(cb, self.closed, self.open)
        listener.state_change(cb, self.open, self.closed)
        closed_to_open = _transitions(name, self.py_closed, self.py_open)
        open_to_closed = _transitions(name, self.py_open, self.py_closed)
        assert closed_to_open == 1
        assert open_to_closed == 1
        listener.state_change(cb, self.closed, self.open)
        second_closed_to_open = _transitions(name, self.py_closed, self.py_open)
        no_second_open_to_closed = _transitions(name, self.py_open, self.py_closed)
        assert second_closed_to_open == 2
        assert no_second_open_to_closed == 1

    def test_unknown_state_raises_key_error(self) -> None:
        """state_change raises KeyError when new_state is not a recognised state name."""
        listener = AsyncCircuitBreakerListener()
        cb = create_async_circuit_breaker("async_unknown_state_test")
        unknown = MagicMock(spec=aiobreaker.state.CircuitBreakerBaseState)
        unknown.state = MagicMock()
        unknown.state.name = "UNKNOWN"
        with pytest.raises(KeyError):
            listener.state_change(cb, self.closed, cast(aiobreaker.state.CircuitBreakerBaseState, unknown))

    def test_recovery_transition_recorded(self) -> None:
        """Open → half_open → closed recovery path is tracked for async breakers."""
        name = "async_recovery_test"
        listener = AsyncCircuitBreakerListener()
        cb = create_async_circuit_breaker(name)
        before_open_to_half = _transitions(name, self.py_open, self.py_half_open)
        before_half_to_closed = _transitions(name, self.py_half_open, self.py_closed)
        listener.state_change(cb, self.open, self.half_open)
        listener.state_change(cb, self.half_open, self.closed)
        after_open_to_half = _transitions(name, self.py_open, self.py_half_open)
        after_half_to_closed = _transitions(name, self.py_half_open, self.py_closed)
        assert after_open_to_half == before_open_to_half + 1
        assert after_half_to_closed == before_half_to_closed + 1
        assert _gauge(name) == 0

    @pytest.mark.asyncio
    async def test_async_breaker_emits_metrics_on_real_transition(self) -> None:
        """Real aiobreaker transitions emit the correct Prometheus metrics.

        Makes sure:
          - After fail_max failures, the breaker opens and the gauge reaches 1.
          - The open transition is counted in transitions_total.
        """
        name = "async_real_transition_test"
        cb = create_async_circuit_breaker(name=name, failure_threshold=2)
        before = _transitions(name, self.py_closed, self.py_open)
        assert cb.state.state.name.lower() == "closed"
        assert _gauge(name) == 0

        async def failing() -> None:
            raise ConnectionError

        # First failure
        with pytest.raises(ConnectionError):
            await cb.call_async(failing)

        # Second failure
        with pytest.raises(aiobreaker.state.CircuitBreakerError):
            await cb.call_async(failing)

        assert cb.state.state.name.lower() == "open"
        assert _gauge(name) == 1.0
        assert _transitions(name, self.py_closed, self.py_open) == before + 1.0

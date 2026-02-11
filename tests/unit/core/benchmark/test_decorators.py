"""Unit tests for timing decorators."""

from __future__ import annotations

import pytest

from core.benchmark.decorators import benchmark_provider, timed


class TestTimed:
    """Tests for @timed decorator."""

    def test_sync_returns_tuple(self):
        """@timed sync function returns (result, elapsed)."""

        @timed
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 3
        assert result[1] >= 0.0

    @pytest.mark.asyncio
    async def test_async_returns_tuple(self):
        """@timed async function returns (result, elapsed)."""

        @timed
        async def add(a, b):
            return a + b

        result = await add(1, 2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 3
        assert result[1] >= 0.0

    def test_sync_elapsed_is_positive(self):
        """Elapsed time is a positive float for sync functions."""

        @timed
        def slow():
            total = 0
            for i in range(10000):
                total += i
            return total

        _, elapsed = slow()
        assert elapsed >= 0.0

    @pytest.mark.asyncio
    async def test_async_elapsed_is_positive(self):
        """Elapsed time is a positive float for async functions."""
        import asyncio

        @timed
        async def slow():
            await asyncio.sleep(0.01)
            return "done"

        result, elapsed = await slow()
        assert result == "done"
        assert elapsed >= 0.01

    def test_preserves_function_name(self):
        """@timed preserves the original function name."""

        @timed
        def my_function():
            return 42

        assert my_function.__name__ == "my_function"


class TestBenchmarkProvider:
    """Tests for @benchmark_provider decorator."""

    @pytest.mark.asyncio
    async def test_async_returns_tuple(self):
        """@benchmark_provider async returns (result, elapsed)."""
        call_count = 0

        @benchmark_provider(warmup=2, cooldown=0.0)
        async def search():
            nonlocal call_count
            call_count += 1
            return "found"

        result = await search()
        assert isinstance(result, tuple)
        assert result[0] == "found"
        assert result[1] >= 0.0

    @pytest.mark.asyncio
    async def test_warmup_call_count(self):
        """Warmup calls happen before the timed measurement."""
        call_count = 0

        @benchmark_provider(warmup=3, cooldown=0.0)
        async def search():
            nonlocal call_count
            call_count += 1
            return "found"

        await search()
        # 3 warmup + 1 measurement = 4 total
        assert call_count == 4

    def test_sync_returns_tuple(self):
        """@benchmark_provider sync returns (result, elapsed)."""
        call_count = 0

        @benchmark_provider(warmup=2, cooldown=0.0)
        def search():
            nonlocal call_count
            call_count += 1
            return "found"

        result = search()
        assert isinstance(result, tuple)
        assert result[0] == "found"
        assert result[1] >= 0.0
        assert call_count == 3  # 2 warmup + 1 measurement

    def test_sync_warmup_call_count(self):
        """Sync warmup calls happen before the timed measurement."""
        call_count = 0

        @benchmark_provider(warmup=5, cooldown=0.0)
        def search():
            nonlocal call_count
            call_count += 1
            return "found"

        search()
        assert call_count == 6  # 5 warmup + 1 measurement

    @pytest.mark.asyncio
    async def test_zero_warmup(self):
        """Zero warmup skips warmup phase."""
        call_count = 0

        @benchmark_provider(warmup=0, cooldown=0.0)
        async def search():
            nonlocal call_count
            call_count += 1
            return "found"

        await search()
        assert call_count == 1  # Only measurement

    def test_preserves_function_name(self):
        """@benchmark_provider preserves the original function name."""

        @benchmark_provider(warmup=1, cooldown=0.0)
        def my_search():
            return 42

        assert my_search.__name__ == "my_search"

    @pytest.mark.asyncio
    async def test_passes_arguments(self):
        """@benchmark_provider passes arguments to the wrapped function."""
        received_args = []

        @benchmark_provider(warmup=1, cooldown=0.0)
        async def search(query, limit=10):
            received_args.append((query, limit))
            return "found"

        await search("red bird", limit=5)
        # 1 warmup + 1 measurement = 2 calls
        assert len(received_args) == 2
        assert all(args == ("red bird", 5) for args in received_args)

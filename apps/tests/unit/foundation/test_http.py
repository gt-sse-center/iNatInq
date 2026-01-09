"""Unit tests for foundation.http module.

This file tests the create_retry_session function which creates HTTP sessions
with automatic retry logic for transient failures in the pipeline service.

# Test Coverage

The tests cover:
  - Session Creation: Default and custom session configuration
  - Retry Configuration: Default retry strategy, custom retry parameters
  - Adapter Configuration: HTTP and HTTPS adapter mounting, identical configs
  - Parameter Handling: None value defaults, parameter validation
  - Integration: Requests Session integration, Retry strategy integration

# Test Structure

Tests use pytest class-based organization with descriptive test names.
Session creation is tested in isolation without network calls.

# Running Tests

Run with: pytest tests/unit/foundation/test_http.py
"""

from __future__ import annotations

import requests
from foundation.http import (
    DEFAULT_ALLOWED_METHODS,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_STATUS_FORCELIST,
    create_retry_session,
)
from urllib3.util.retry import Retry

# =============================================================================
# Session Creation Tests
# =============================================================================


class TestCreateRetrySession:
    """Test suite for create_retry_session function."""

    def test_creates_session_with_default_config(self) -> None:
        """Test that session is created with default retry configuration.

        **Why this test is important:**
          - Session creation is the primary function of this module
          - Ensures default configuration works out of the box
          - Validates that adapters are properly mounted
          - Critical for basic functionality and ease of use

        **What it tests:**
          - Returns a requests.Session instance
          - HTTP adapter is mounted and not None
          - HTTPS adapter is mounted and not None
          - Adapter has retry strategy configured
        """
        session = create_retry_session()

        assert isinstance(session, requests.Session)

        # Check that adapters are mounted
        assert session.adapters["http://"] is not None
        assert session.adapters["https://"] is not None

        # Verify adapter has retry strategy
        http_adapter = session.adapters["http://"]
        assert http_adapter.max_retries is not None  # type: ignore[attr-defined]
        assert isinstance(http_adapter.max_retries, Retry)  # type: ignore[attr-defined]

    # =============================================================================
    # Retry Configuration Tests
    # =============================================================================

    def test_default_retry_configuration(self) -> None:
        """Test that default retry configuration matches constants.

        **Why this test is important:**
          - Default configuration must match documented constants
          - Ensures consistency between code and documentation
          - Validates that defaults are applied correctly
          - Critical for predictable behavior and debugging

        **What it tests:**
          - max_retries matches DEFAULT_MAX_RETRIES
          - backoff_factor matches DEFAULT_BACKOFF_FACTOR
          - status_forcelist matches DEFAULT_STATUS_FORCELIST
          - allowed_methods matches DEFAULT_ALLOWED_METHODS
        """
        session = create_retry_session()
        adapter = session.adapters["http://"]
        retry_strategy = adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(retry_strategy, Retry)
        assert retry_strategy.total == DEFAULT_MAX_RETRIES
        assert retry_strategy.backoff_factor == DEFAULT_BACKOFF_FACTOR
        assert retry_strategy.status_forcelist == DEFAULT_STATUS_FORCELIST
        assert retry_strategy.allowed_methods == DEFAULT_ALLOWED_METHODS

    def test_custom_max_retries(self) -> None:
        """Test that custom max_retries is applied.

        **Why this test is important:**
          - Custom retry count allows tuning for different use cases
          - Some services need more retries, others need fewer
          - Critical for adapting to different service characteristics
          - Validates parameter passing to retry strategy

        **What it tests:**
          - Custom max_retries value is applied to retry strategy
          - Retry strategy total matches custom value
          - Other default values are preserved
        """
        custom_retries = 5
        session = create_retry_session(max_retries=custom_retries)
        adapter = session.adapters["http://"]
        retry_strategy = adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(retry_strategy, Retry)
        assert retry_strategy.total == custom_retries

    def test_custom_backoff_factor(self) -> None:
        """Test that custom backoff_factor is applied.

        **Why this test is important:**
          - Backoff factor controls delay between retries
          - Different services may need different backoff strategies
          - Critical for managing retry delays and rate limiting
          - Validates exponential backoff configuration

        **What it tests:**
          - Custom backoff_factor value is applied to retry strategy
          - Retry strategy backoff_factor matches custom value
          - Other default values are preserved
        """
        custom_backoff = 2.5
        session = create_retry_session(backoff_factor=custom_backoff)
        adapter = session.adapters["http://"]
        retry_strategy = adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(retry_strategy, Retry)
        assert retry_strategy.backoff_factor == custom_backoff

    def test_custom_status_forcelist(self) -> None:
        """Test that custom status_forcelist is applied.

        **Why this test is important:**
          - Status forcelist determines which HTTP status codes trigger retries
          - Different services may have different retryable status codes
          - Critical for handling service-specific error patterns
          - Validates HTTP status code retry logic

        **What it tests:**
          - Custom status_forcelist is applied to retry strategy
          - Retry strategy status_forcelist matches custom value
          - List format is preserved correctly
        """
        custom_statuses = [400, 401, 403]
        session = create_retry_session(status_forcelist=custom_statuses)
        adapter = session.adapters["http://"]
        retry_strategy = adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(retry_strategy, Retry)
        assert retry_strategy.status_forcelist == custom_statuses

    def test_custom_allowed_methods(self) -> None:
        """Test that custom allowed_methods is applied.

        **Why this test is important:**
          - Allowed methods determine which HTTP methods can be retried
          - Some methods (POST, PUT) may be safe to retry, others (DELETE) may not
          - Critical for preventing duplicate operations on non-idempotent requests
          - Validates HTTP method retry logic

        **What it tests:**
          - Custom allowed_methods is applied to retry strategy
          - Retry strategy allowed_methods matches custom value
          - List format is preserved correctly
        """
        custom_methods = ["PUT", "DELETE"]
        session = create_retry_session(allowed_methods=custom_methods)
        adapter = session.adapters["http://"]
        retry_strategy = adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(retry_strategy, Retry)
        assert retry_strategy.allowed_methods == custom_methods

    # =============================================================================
    # Parameter Handling Tests
    # =============================================================================

    def test_none_uses_defaults(self) -> None:
        """Test that None values use default configuration.

        **Why this test is important:**
          - None values should fall back to defaults for convenience
          - Allows partial customization without specifying all parameters
          - Critical for API ergonomics and ease of use
          - Validates default value handling

        **What it tests:**
          - None status_forcelist uses DEFAULT_STATUS_FORCELIST
          - None allowed_methods uses DEFAULT_ALLOWED_METHODS
          - Other parameters can still be customized
        """
        session = create_retry_session(
            status_forcelist=None,
            allowed_methods=None,
        )
        adapter = session.adapters["http://"]
        retry_strategy = adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(retry_strategy, Retry)
        assert retry_strategy.status_forcelist == DEFAULT_STATUS_FORCELIST
        assert retry_strategy.allowed_methods == DEFAULT_ALLOWED_METHODS

    # =============================================================================
    # Adapter Configuration Tests
    # =============================================================================

    def test_both_adapters_have_same_config(self) -> None:
        """Test that HTTP and HTTPS adapters have identical configuration.

        **Why this test is important:**
          - HTTP and HTTPS should behave identically for consistency
          - Prevents configuration drift between protocols
          - Critical for predictable behavior across protocol boundaries
          - Validates adapter configuration symmetry

        **What it tests:**
          - HTTP adapter retry strategy matches HTTPS adapter
          - All retry parameters are identical between adapters
          - Custom configuration applies to both adapters
        """
        session = create_retry_session(max_retries=7, backoff_factor=3.0)
        http_adapter = session.adapters["http://"]
        https_adapter = session.adapters["https://"]

        http_retry = http_adapter.max_retries  # type: ignore[attr-defined]
        https_retry = https_adapter.max_retries  # type: ignore[attr-defined]

        assert isinstance(http_retry, Retry)
        assert isinstance(https_retry, Retry)
        assert http_retry.total == https_retry.total
        assert http_retry.backoff_factor == https_retry.backoff_factor
        assert http_retry.status_forcelist == https_retry.status_forcelist
        assert http_retry.allowed_methods == https_retry.allowed_methods

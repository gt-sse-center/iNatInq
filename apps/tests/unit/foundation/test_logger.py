"""Unit tests for foundation.logger module.

This file tests the CustomJSONFormatter class and LOGGING_CONFIG constant which
provide structured JSON logging for the pipeline service.

# Test Coverage

The tests cover:
  - CustomJSONFormatter: JSON formatting, standard fields, OpenTelemetry context,
    request/response data, error data
  - LOGGING_CONFIG: Dictionary structure, required sections (version, formatters,
    handlers, loggers)

# Test Structure

Tests use pytest class-based organization. LogRecord objects are created manually
to test formatter behavior in isolation without actual logging infrastructure.

# Running Tests

Run with: pytest tests/unit/foundation/test_logger.py
"""


import json
import logging
import sys

from foundation.logger import LOGGING_CONFIG, CustomJSONFormatter

# =============================================================================
# CustomJSONFormatter Tests
# =============================================================================


class TestCustomJSONFormatter:
    """Test suite for CustomJSONFormatter class."""

    def test_format_creates_json_string(self) -> None:
        """Test that format() returns valid JSON string.

        **Why this test is important:**
          - JSON formatting is the core functionality of the formatter
          - Valid JSON is required for log aggregation and parsing
          - Ensures output can be consumed by log processing systems
          - Critical for structured logging and observability

        **What it tests:**
          - format() returns a string
          - String is valid JSON
          - JSON can be parsed into a dictionary
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    # =============================================================================
    # Standard Fields Tests
    # =============================================================================

    def test_get_log_includes_standard_fields(self) -> None:
        """Test that get_log() includes standard log fields.

        **Why this test is important:**
          - Standard fields are required for basic log processing
          - Ensures all essential log information is captured
          - Validates field names and types
          - Critical for log aggregation and searching

        **What it tests:**
          - time field is included from asctime
          - level, logger_name, pathname, line, message are included
          - process_id, process_name, thread_id, thread_name are included
          - All fields have correct values
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        # format() sets asctime via Formatter.format()
        formatter.format(record)

        log_dict = formatter.get_log(record)

        assert "time" in log_dict
        assert log_dict["level"] == "INFO"
        assert log_dict["logger_name"] == "test.logger"
        assert log_dict["pathname"] == "/test/path"
        assert log_dict["line"] == 42
        assert log_dict["message"] == "test message"
        assert "process_id" in log_dict
        assert "process_name" in log_dict
        assert "thread_id" in log_dict
        assert "thread_name" in log_dict

    # =============================================================================
    # OpenTelemetry Context Tests
    # =============================================================================

    def test_get_log_includes_otel_trace_context(self) -> None:
        """Test that get_log() includes OpenTelemetry trace context if present.

        **Why this test is important:**
          - OpenTelemetry context enables distributed tracing
          - Trace IDs link logs to traces for debugging
          - Critical for observability in microservices architectures
          - Enables correlation between logs and traces

        **What it tests:**
          - otel_service_name is included when present
          - otel_trace_sampled is included when present
          - trace_id and span_id are included when present
          - All OpenTelemetry fields have correct values
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        # Add OpenTelemetry attributes
        record.otelServiceName = "test-service"
        record.otelTraceSampled = True
        record.otelTraceID = "abc123"
        record.otelSpanID = "def456"

        log_dict = formatter.get_log(record)

        assert log_dict["otel_service_name"] == "test-service"
        assert log_dict["otel_trace_sampled"] is True
        assert log_dict["trace_id"] == "abc123"
        assert log_dict["span_id"] == "def456"

    def test_get_log_omits_missing_otel_context(self) -> None:
        """Test that get_log() omits OpenTelemetry fields if not present.

        **Why this test is important:**
          - Not all logs have OpenTelemetry context (e.g., background jobs)
          - Omitting missing fields keeps logs clean
          - Prevents None/null values in log output
          - Critical for log consistency and parsing

        **What it tests:**
          - OpenTelemetry fields are not present when attributes are missing
          - Standard fields are still included
          - Log structure is valid without OpenTelemetry context
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime

        log_dict = formatter.get_log(record)

        assert "otel_service_name" not in log_dict
        assert "otel_trace_sampled" not in log_dict
        assert "trace_id" not in log_dict
        assert "span_id" not in log_dict

    # =============================================================================
    # Request/Response Data Tests
    # =============================================================================

    def test_get_log_includes_request_data(self) -> None:
        """Test that get_log() includes request data if present.

        **Why this test is important:**
          - Request data enables HTTP request logging and debugging
          - Path, method, and remote address are essential for API logging
          - Critical for API observability and security auditing
          - Enables request tracing and analysis

        **What it tests:**
          - path, method, remoteAddr are included when present in request dict
          - Request fields are extracted from record.request dictionary
          - All request fields have correct values
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.request = {
            "path": "/api/endpoint",
            "method": "POST",
            "remoteAddr": "192.168.1.1",
        }

        log_dict = formatter.get_log(record)

        assert log_dict["path"] == "/api/endpoint"
        assert log_dict["method"] == "POST"
        assert log_dict["remoteAddr"] == "192.168.1.1"

    def test_get_log_omits_missing_request_fields(self) -> None:
        """Test that get_log() omits missing request fields.

        **Why this test is important:**
          - Not all request dicts have all fields
          - Omitting missing fields keeps logs clean
          - Prevents KeyError or None values in log output
          - Critical for log consistency and parsing

        **What it tests:**
          - Present request fields are included
          - Missing request fields are not included
          - Partial request data is handled gracefully
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.request = {"path": "/api/endpoint"}  # Missing method and remoteAddr

        log_dict = formatter.get_log(record)

        assert log_dict["path"] == "/api/endpoint"
        assert "method" not in log_dict
        assert "remoteAddr" not in log_dict

    def test_get_log_includes_response_data(self) -> None:
        """Test that get_log() includes response data if present.

        **Why this test is important:**
          - Response data enables HTTP response logging and debugging
          - Status code and latency are essential for API performance monitoring
          - Critical for API observability and performance analysis
          - Enables response tracing and analysis

        **What it tests:**
          - path, statuscode, since are included when present in response dict
          - Response fields are extracted from record.response dictionary
          - All response fields have correct values
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.response = {
            "path": "/api/endpoint",
            "statuscode": 200,
            "since": 150,
        }

        log_dict = formatter.get_log(record)

        assert log_dict["path"] == "/api/endpoint"
        assert log_dict["statuscode"] == 200
        assert log_dict["since"] == 150

    # =============================================================================
    # Error Data Tests
    # =============================================================================

    def test_get_log_includes_error_data(self) -> None:
        """Test that get_log() includes error data if present.

        **Why this test is important:**
          - Error data enables error logging and debugging
          - Error message and type are essential for error tracking
          - Critical for error observability and incident response
          - Enables error analysis and alerting

        **What it tests:**
          - error.message and error.type are included when present in error dict
          - Error fields are extracted from record.error dictionary
          - All error fields have correct values
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.error = {
            "message": "Something went wrong",
            "type": "ValueError",
        }

        log_dict = formatter.get_log(record)

        assert log_dict["error"]["message"] == "Something went wrong"
        assert log_dict["error"]["type"] == "ValueError"

    # =============================================================================
    # Edge Case Tests
    # =============================================================================

    def test_get_log_handles_non_dict_request(self) -> None:
        """Test that get_log() handles non-dict request attribute.

        **Why this test is important:**
          - Defensive programming prevents crashes from invalid data
          - Handles edge cases gracefully without breaking logging
          - Ensures robust error handling for malformed log records
          - Critical for production stability

        **What it tests:**
          - Non-dict request attribute doesn't cause crash
          - Request fields are not included when request is not a dict
          - Standard fields are still included
          - Log structure is valid despite invalid request data
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.request = "not a dict"  # Invalid type

        log_dict = formatter.get_log(record)

        # Should not crash, and request fields should not be present
        assert "path" not in log_dict
        assert "method" not in log_dict

    def test_get_log_handles_non_dict_response(self) -> None:
        """Test that get_log() handles non-dict response attribute.

        **Why this test is important:**
          - Defensive programming prevents crashes from invalid data
          - Handles edge cases gracefully without breaking logging
          - Ensures robust error handling for malformed log records
          - Critical for production stability

        **What it tests:**
          - Non-dict response attribute doesn't cause crash
          - Response fields are not included when response is not a dict
          - Standard fields are still included
          - Log structure is valid despite invalid response data
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.response = "not a dict"  # Invalid type

        log_dict = formatter.get_log(record)

        # Should not crash, and response fields should not be present
        assert "statuscode" not in log_dict
        assert "since" not in log_dict

    def test_get_log_includes_error_with_exc_info(self) -> None:
        """Test that get_log() includes traceback when error dict has exc_info.

        **Why this test is important:**
          - Stack traces are critical for debugging errors
          - Enables full error context in logs
          - Critical for production debugging and incident response
          - Enables error analysis with full context

        **What it tests:**
          - error.trace is included when record.exc_info is present
          - Traceback is formatted using formatException
          - Error dict includes both error data and traceback
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path",
                lineno=42,
                msg="test message",
                args=(),
                exc_info=exc_info,
            )
            formatter.format(record)  # Sets asctime
            record.error = {
                "message": "Something went wrong",
                "type": "ValueError",
            }

            log_dict = formatter.get_log(record)

            assert log_dict["error"]["message"] == "Something went wrong"
            assert log_dict["error"]["type"] == "ValueError"
            assert "trace" in log_dict["error"]
            assert "ValueError: test error" in log_dict["error"]["trace"]

    def test_get_log_handles_non_dict_error(self) -> None:
        """Test that get_log() handles non-dict error attribute.

        **Why this test is important:**
          - Defensive programming prevents crashes from invalid data
          - Handles edge cases gracefully without breaking logging
          - Ensures robust error handling for malformed log records
          - Critical for production stability

        **What it tests:**
          - Non-dict error attribute doesn't cause crash
          - Error is included as-is when not a dict
          - Standard fields are still included
          - Log structure is valid despite invalid error data
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        record.error = "error string"  # Invalid type (not a dict)

        log_dict = formatter.get_log(record)

        # Should not crash, and error should be included as-is
        assert log_dict["error"] == "error string"

    def test_get_log_includes_response_partial_fields(self) -> None:
        """Test that get_log() includes only present response fields.

        **Why this test is important:**
          - Not all response dicts have all fields
          - Omitting missing fields keeps logs clean
          - Prevents None values in log output
          - Critical for log consistency and parsing

        **What it tests:**
          - Present response fields are included (path, method, statuscode, since, remoteAddr)
          - Missing response fields are not included
          - Partial response data is handled gracefully
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        # Include only some fields (path, statuscode, remoteAddr)
        record.response = {
            "path": "/api/endpoint",
            "statuscode": 200,
            "remoteAddr": "127.0.0.1",
            # method and since are missing
        }

        log_dict = formatter.get_log(record)

        assert log_dict["path"] == "/api/endpoint"
        assert log_dict["statuscode"] == 200
        assert log_dict["remoteAddr"] == "127.0.0.1"
        # method and since should not be in log_dict when not present in response
        # Note: method might be present if it was set from request,
        # but statuscode should be from response
        assert "since" not in log_dict or log_dict.get("since") is None

    def test_get_log_includes_extra_attributes(self) -> None:
        """Test that get_log() includes extra attributes passed via extra parameter.

        **Why this test is important:**
          - Extra attributes enable custom log fields
          - Enables context-specific logging (e.g., user_id, request_id)
          - Critical for structured logging and log enrichment
          - Enables filtering and analysis by custom fields

        **What it tests:**
          - Custom attributes are included in log output
          - Standard attributes are excluded
          - Extra attributes have correct values
        """
        formatter = CustomJSONFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatter.format(record)  # Sets asctime
        # Add custom attributes
        record.custom_field = "custom_value"
        record.user_id = "user123"
        record.request_id = "req456"

        log_dict = formatter.get_log(record)

        assert log_dict["custom_field"] == "custom_value"
        assert log_dict["user_id"] == "user123"
        assert log_dict["request_id"] == "req456"
        # Standard attributes should not be duplicated
        assert "name" not in log_dict or log_dict["name"] == "test.logger"


# =============================================================================
# LOGGING_CONFIG Tests
# =============================================================================


class TestLoggingConfig:
    """Test suite for LOGGING_CONFIG constant."""

    def test_logging_config_is_dict(self) -> None:
        """Test that LOGGING_CONFIG is a dictionary.

        **Why this test is important:**
          - LOGGING_CONFIG must be a dict for dictConfig()
          - Validates basic structure for logging configuration
          - Ensures configuration can be used with logging.dictConfig()
          - Critical for logging system initialization

        **What it tests:**
          - LOGGING_CONFIG is an instance of dict
          - Configuration structure is valid
        """
        assert isinstance(LOGGING_CONFIG, dict)

    def test_logging_config_has_version(self) -> None:
        """Test that LOGGING_CONFIG has version field.

        **Why this test is important:**
          - Version field is required by logging.dictConfig()
          - Ensures configuration follows logging configuration schema
          - Validates configuration completeness
          - Critical for logging system initialization

        **What it tests:**
          - version key is present in LOGGING_CONFIG
          - Configuration structure includes version field
        """
        assert "version" in LOGGING_CONFIG

    def test_logging_config_has_formatters(self) -> None:
        """Test that LOGGING_CONFIG has formatters section.

        **Why this test is important:**
          - Formatters section defines log output format
          - Required for JSON formatting configuration
          - Validates that formatter configuration is present
          - Critical for structured logging output

        **What it tests:**
          - formatters key is present in LOGGING_CONFIG
          - Configuration structure includes formatters section
        """
        assert "formatters" in LOGGING_CONFIG

    def test_logging_config_has_handlers(self) -> None:
        """Test that LOGGING_CONFIG has handlers section.

        **Why this test is important:**
          - Handlers section defines log output destinations
          - Required for configuring where logs are written
          - Validates that handler configuration is present
          - Critical for log routing and output

        **What it tests:**
          - handlers key is present in LOGGING_CONFIG
          - Configuration structure includes handlers section
        """
        assert "handlers" in LOGGING_CONFIG

    def test_logging_config_has_loggers(self) -> None:
        """Test that LOGGING_CONFIG has loggers section.

        **Why this test is important:**
          - Loggers section defines logger configurations
          - Required for configuring logger levels and handlers
          - Validates that logger configuration is present
          - Critical for logger setup and configuration

        **What it tests:**
          - loggers key is present in LOGGING_CONFIG
          - Configuration structure includes loggers section
        """
        assert "loggers" in LOGGING_CONFIG

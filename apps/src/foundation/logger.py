"""Logging configuration with structured JSON formatter.

This module provides a custom JSON formatter and logging configuration
that matches the pattern used in the smarts service (services-main).
"""

import json
import logging
from typing import Any


class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging.

    This formatter converts log records to JSON format, including:
    - Standard log fields (time, level, message, etc.)
    - Request/response information from LoggerMiddleware
    - Error information from ExceptionHandlerMiddleware
    - OpenTelemetry trace context (if LoggingInstrumentor is used)
    - All extra attributes passed via the extra parameter
    """

    def __init__(self, fmt: str) -> None:
        """Initialize the formatter.

        Args:
            fmt: Format string (used for asctime, but output is JSON).
        """
        logging.Formatter.__init__(self, fmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        logging.Formatter.format(self, record)
        return json.dumps(self.get_log(record), indent=None)

    def get_log(self, record: logging.LogRecord) -> dict[str, Any]:
        """Extract log data from record into a dictionary.

        Args:
            record: The log record to extract data from.

        Returns:
            Dictionary containing log data.
        """
        d: dict[str, Any] = {
            "time": record.asctime,
            "process_name": record.processName,
            "process_id": record.process,
            "thread_name": record.threadName,
            "thread_id": record.thread,
            "level": record.levelname,
            "logger_name": record.name,
            "pathname": record.pathname,
            "line": record.lineno,
            "message": record.message,
        }

        # Add OpenTelemetry trace context if present (from LoggingInstrumentor)
        otel_service_name = getattr(record, "otelServiceName", None)
        if otel_service_name is not None:
            d["otel_service_name"] = otel_service_name

        otel_trace_sampled = getattr(record, "otelTraceSampled", None)
        if otel_trace_sampled is not None:
            d["otel_trace_sampled"] = otel_trace_sampled

        otel_trace_id = getattr(record, "otelTraceID", None)
        if otel_trace_id is not None:
            d["trace_id"] = otel_trace_id

        otel_span_id = getattr(record, "otelSpanID", None)
        if otel_span_id is not None:
            d["span_id"] = otel_span_id

        # Add request information if present (from LoggerMiddleware)
        request_data = getattr(record, "request", None)
        if request_data is not None:
            if isinstance(request_data, dict):
                path = request_data.get("path")
                method = request_data.get("method")
                remote_addr = request_data.get("remoteAddr")
                if path is not None:
                    d["path"] = path
                if method is not None:
                    d["method"] = method
                if remote_addr is not None:
                    d["remoteAddr"] = remote_addr

        # Add response information if present (from LoggerMiddleware)
        response_data = getattr(record, "response", None)
        if response_data is not None:
            if isinstance(response_data, dict):
                path = response_data.get("path")
                method = response_data.get("method")
                statuscode = response_data.get("statuscode")
                since = response_data.get("since")
                remote_addr = response_data.get("remoteAddr")
                if path is not None:
                    d["path"] = path
                if method is not None:
                    d["method"] = method
                if statuscode is not None:
                    d["statuscode"] = statuscode
                if since is not None:
                    d["since"] = since
                if remote_addr is not None:
                    d["remoteAddr"] = remote_addr

        # Add error information if present (from ExceptionHandlerMiddleware)
        error_data = getattr(record, "error", None)
        if error_data is not None:
            if isinstance(error_data, dict):
                error_dict: dict[str, Any] = error_data.copy()
                if record.exc_info:
                    error_dict["trace"] = self.formatException(record.exc_info)
                d["error"] = error_dict
            else:
                d["error"] = error_data

        # Add all extra attributes passed via extra parameter
        # Standard LogRecord attributes to exclude (already handled above or internal)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "asctime",
            "request",
            "response",
            "error",
            "otelServiceName",
            "otelTraceSampled",
            "otelTraceID",
            "otelSpanID",
        }
        # Include all non-standard attributes (from extra parameter)
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                d[key] = value

        return d


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Keep existing loggers, just configure them
    "formatters": {
        "standard": {"()": lambda: CustomJSONFormatter(fmt="%(asctime)s")},
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "WARNING",  # Suppress default access logs, use LoggerMiddleware instead
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.asgi": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "pipeline.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "pipeline.error": {
            "handlers": ["default"],
            "level": "ERROR",
            "propagate": False,
        },
        "pipeline.spark": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["default"],
        "level": "INFO",
    },
}

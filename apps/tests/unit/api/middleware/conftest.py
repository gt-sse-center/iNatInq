"""Shared logging fixtures for middleware tests."""

import logging

import pytest


@pytest.fixture(autouse=True)
def attach_caplog_handler(caplog):
    """Attach pytest's caplog handler to middleware loggers.

    The production logger config uses non-propagating loggers, so tests need
    to attach caplog's handler directly to capture records.
    """
    logger_names = ("pipeline.access", "uvicorn.access", "uvicorn.error")
    attached = []
    for name in logger_names:
        logger = logging.getLogger(name)
        if caplog.handler not in logger.handlers:
            logger.addHandler(caplog.handler)
            attached.append(logger)

    try:
        yield
    finally:
        for logger in attached:
            logger.removeHandler(caplog.handler)

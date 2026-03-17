"""Unit tests for the DLQ class.

How to run:
    uv run pytest tests/unit/foundation/dlq/test_dlq.py
"""

import pytest
from unittest.mock import MagicMock
from foundation.dead_letter_queue.dlq import DLQ


def mock_dlq_backend() -> MagicMock:
    mock = MagicMock()
    mock.insert = MagicMock()
    mock.name = "mock-dlq-backend"
    return mock


class TestDLQ:
    def test_enqueue_inserts_into_backend(self):
        mock_backend = mock_dlq_backend()
        dlq = DLQ(mock_backend)

        image_id = "image_123"
        metadata: dict[str, object] = {"reason": "embedding failed", "retries": 2}
        dlq.enqueue_failed_ingestion(image_id, metadata=metadata)

        mock_backend.insert.assert_called_once_with(image_id, metadata=metadata)

    def test_enqueue_logs_errors(self, caplog: pytest.LogCaptureFixture):
        mock_backend = mock_dlq_backend()
        mock_backend.insert.side_effect = Exception("uh oh")
        dlq = DLQ(mock_backend)

        image_id = "image_123"
        metadata: dict[str, object] = {"reason": "embedding failed", "retries": 2}

        with caplog.at_level("ERROR"):
            dlq.enqueue_failed_ingestion(image_id, metadata=metadata)

        assert f"Failed to insert {image_id}" in caplog.text
        assert "uh oh" in caplog.text

"""Unit tests for the DELETE /cache endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.app import create_app


def test_delete_cache_endpoint_success() -> None:
    """DELETE /cache returns 200 and invalidates the cache.

    Why important: Confirms the manual cache-bust path works end-to-end
    and that the ``invalidate()`` method is actually invoked.
    """
    mock_cache = MagicMock()
    mock_cache.invalidate = AsyncMock()

    with (
        patch("api.app.get_semantic_cache", return_value=None),
        patch("api.routes.get_semantic_cache", return_value=mock_cache),
    ):
        app = create_app()
        client = TestClient(app)
        resp = client.delete("/cache")

    assert resp.status_code == 200
    assert resp.json() == {"status": "cache invalidated"}
    mock_cache.invalidate.assert_awaited_once()


def test_delete_cache_endpoint_disabled() -> None:
    """DELETE /cache returns 400 when the cache is not enabled.

    Why important: The endpoint must not silently succeed when caching
    is disabled — callers need clear feedback.
    """
    with (
        patch("api.app.get_semantic_cache", return_value=None),
        patch("api.routes.get_semantic_cache", return_value=None),
    ):
        app = create_app()
        client = TestClient(app)
        resp = client.delete("/cache")

    assert resp.status_code == 400

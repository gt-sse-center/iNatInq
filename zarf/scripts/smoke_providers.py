#!/usr/bin/env python3
"""Smoke test for embedding + vector DB providers.

This script performs a minimal, end-to-end verification:
1) Generate an embedding for a small text input.
2) Upsert the embedding into the configured vector database.
3) Search the same collection using the same embedding.

If all three steps succeed, you can be confident the embedding and vector
database providers are wired correctly in this environment.

Usage:
  python zarf/scripts/smoke_providers.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

logger = logging.getLogger(__name__)
_IMPORTS = None


def _init_repo_imports() -> None:
    """Ensure repo modules are importable and load them once."""
    global _IMPORTS  # noqa: PLW0603
    if _IMPORTS is not None:
        return

    # Resolve the repo root by walking up from this file:
    # zarf/scripts/smoke_providers.py -> repo root (parents[2]).
    repo_root = Path(__file__).resolve().parents[2]
    # The Python package modules live under src/ at the repo root.
    src_path = repo_root / "src"
    # Only insert if missing to avoid duplicating paths.
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from clients.interfaces.embedding import create_embedding_provider
    from clients.interfaces.vector_db import create_vector_db_provider
    from config import VectorDBConfig, get_settings
    from core.models import VectorPoint

    _IMPORTS = SimpleNamespace(
        create_embedding_provider=create_embedding_provider,
        create_vector_db_provider=create_vector_db_provider,
        VectorDBConfig=VectorDBConfig,
        get_settings=get_settings,
        VectorPoint=VectorPoint,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI args to allow light customization of the smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test external providers.")
    # Provider override for testing a specific backend even if env default differs.
    parser.add_argument(
        "--provider",
        choices=("qdrant",),
        help="Override VECTOR_DB_PROVIDER for this run.",
    )
    # Optional collection name to avoid collisions or to re-use a known target.
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name (defaults to smoke_<random>; auto-cleaned if omitted).",
    )
    # Input text to embed; useful for quick variation or debugging.
    parser.add_argument(
        "--text",
        default="smoke test",
        help="Text to embed and upsert.",
    )
    return parser.parse_args()


def _build_points(text: str, vector: list[float]):
    """Create VectorPoint objects for upsert."""
    _init_repo_imports()
    assert _IMPORTS is not None

    return [
        _IMPORTS.VectorPoint(
            # Qdrant allows string IDs; we generate a unique one for test data.
            id=str(uuid4()),
            # Vector payload uses the embedding directly.
            vector=vector,
            # Store raw text in payload for easy inspection later.
            payload={"text": text},
        )
    ]


def _print_provider_env() -> None:
    """Print provider-related env vars, masking secrets."""
    logger.info("External provider config from environment:")
    logger.info("  VECTOR_DB_PROVIDER=%s", os.getenv("VECTOR_DB_PROVIDER", ""))
    logger.info("  QDRANT_URL=%s", os.getenv("QDRANT_URL", ""))
    has_key = os.getenv("QDRANT_API_KEY") is not None
    logger.info("  QDRANT_API_KEY=%s", "(set)" if has_key else "(unset)")


def _configure_logging() -> None:
    """Configure logging for the smoke test script."""
    logging.basicConfig(level=logging.INFO, format="[smoke-providers] %(message)s")


async def _cleanup_collection(provider_type: str, vector_db: object, collection: str) -> None:
    """Best-effort cleanup for smoke test collections."""
    try:
        if provider_type == "qdrant":
            qdrant_client = vector_db.client
            if qdrant_client is None:
                logger.info("Cleanup skipped: Qdrant client unavailable")
                return
            logger.info("Cleaning up collection=%s", collection)
            await qdrant_client.delete_collection(collection_name=collection)
            logger.info("Cleanup completed")
            return
        logger.info("Cleanup skipped: unsupported provider_type=%s", provider_type)
    except Exception as exc:
        logger.info("Cleanup failed for collection=%s: %s", collection, exc)


async def _wait_for_collection(
    provider_type: str,
    vector_db: object,
    collection: str,
    timeout_s: float = 30.0,
    interval_s: float = 0.5,
) -> None:
    """Wait until the collection is visible to the provider."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if provider_type == "qdrant":
            qdrant_client = vector_db.client
            if qdrant_client is None:
                break
            collections = await qdrant_client.get_collections()
            existing = {c.name for c in collections.collections}
            if collection in existing:
                return
        else:
            break
        await asyncio.sleep(interval_s)
    raise RuntimeError(f"Collection not ready after {timeout_s:.1f}s: {collection}")


async def _run_smoke_test(
    provider_type: str,
    collection: str,
    text: str,
    *,
    cleanup: bool,
) -> int:
    """Execute the smoke test steps in an async flow."""
    _init_repo_imports()
    assert _IMPORTS is not None

    # Load settings from environment (e.g., .env.local or compose env).
    settings = _IMPORTS.get_settings()
    _print_provider_env()
    logger.info("Using embedding provider: %s", settings.embedding.provider_type)
    # Build the embedding provider using configured settings.
    embedder = _IMPORTS.create_embedding_provider(settings.embedding)

    # Generate the embedding vector for the supplied text.
    logger.info("Requesting embedding from provider")
    try:
        vector = embedder.embed(text)
    except Exception as exc:
        provider_name = settings.embedding.provider_type
        raise RuntimeError(f"Embedding request failed for provider={provider_name}") from exc
    vector_size = len(vector)
    if not vector_size:
        raise RuntimeError("Embedding provider returned an empty vector.")
    logger.info("Embedding returned with vector_size=%s", vector_size)

    # Restrict to providers supported by the compose/dev setup.
    if provider_type != "qdrant":
        raise RuntimeError(f"Unsupported provider type: {provider_type}")

    # Use settings' vector DB config when it matches the requested provider.
    # Otherwise build a provider-specific config from environment.
    if provider_type == settings.vector_db.provider_type:
        vector_cfg = settings.vector_db
    else:
        vector_cfg = _IMPORTS.VectorDBConfig.from_env_for_provider(
            provider_type=provider_type,
        )

    logger.info("Using vector DB provider: %s", provider_type)
    logger.info("Qdrant URL: %s", vector_cfg.qdrant_url)
    # Instantiate the vector DB provider.
    vector_db = _IMPORTS.create_vector_db_provider(vector_cfg)
    # Build points for upsert.
    points = _build_points(text, vector)

    try:
        # Upsert the test point (ensures the collection exists as needed).
        logger.info("Upserting 1 point into collection=%s", collection)
        await vector_db.batch_upsert_async(
            collection=collection,
            points=points,
            vector_size=vector_size,
        )
        logger.info("Upsert completed")
        logger.info("Waiting for collection=%s to be ready", collection)
        await _wait_for_collection(provider_type, vector_db, collection)
        logger.info("Collection is ready")
        # Search with the same vector to confirm the DB returns results.
        logger.info("Searching collection=%s with limit=1", collection)
        results = await vector_db.search_async(
            collection=collection,
            query_vector=vector,
            limit=1,
        )
        logger.info("Search completed with total=%s", results.total)

        # Print a compact summary for humans and CI logs.
        logger.info(
            "provider: %s collection: %s vector_size: %s results: %s",
            provider_type,
            collection,
            vector_size,
            results.total,
        )
        return 0
    finally:
        if cleanup:
            await _cleanup_collection(provider_type, vector_db, collection)


def main() -> int:
    """Entry point for CLI execution."""
    _configure_logging()
    _init_repo_imports()
    # Read CLI arguments.
    args = _parse_args()

    # Determine provider type:
    # - CLI flag if provided
    # - Otherwise, the default configured in environment/settings
    provider_type = args.provider
    # Create a unique collection if none is provided.
    collection = args.collection or f"smoke_{uuid4().hex[:8]}"
    cleanup = args.collection is None
    # Text to embed and store.
    text = args.text

    if provider_type is None:
        assert _IMPORTS is not None

        # Default to the configured provider when no override is given.
        provider_type = _IMPORTS.get_settings().vector_db.provider_type

    # Run the async smoke test and return its exit code.
    return asyncio.run(_run_smoke_test(provider_type, collection, text, cleanup=cleanup))


if __name__ == "__main__":
    # Ensure a proper exit code is returned to the shell.
    raise SystemExit(main())

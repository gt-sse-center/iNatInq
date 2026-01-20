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
  python zarf/scripts/smoke_providers.py --provider weaviate
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def _add_repo_src_to_path() -> None:
    """Ensure repo src/ is importable when running from the workspace root.

    This script lives under zarf/scripts/, so we add the repo's src/ directory
    to sys.path to import application modules without requiring installation.
    """
    # Resolve the repo root by walking up from this file:
    # zarf/scripts/smoke_providers.py -> repo root (parents[2]).
    repo_root = Path(__file__).resolve().parents[2]
    # The Python package modules live under src/ at the repo root.
    src_path = repo_root / "src"
    # Only insert if missing to avoid duplicating paths.
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _parse_args() -> argparse.Namespace:
    """Parse CLI args to allow light customization of the smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test external providers.")
    # Provider override for testing a specific backend even if env default differs.
    parser.add_argument(
        "--provider",
        choices=("qdrant", "weaviate"),
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


def _build_points(provider_type: str, text: str, vector: list[float]):
    """Create provider-specific point objects for upsert.

    Qdrant uses VectorPoint (our wrapper), while Weaviate uses WeaviateDataObject.
    """
    # Weaviate expects objects with explicit UUIDs and property dictionaries.
    if provider_type == "weaviate":
        from clients.weaviate import WeaviateDataObject

        return [
            WeaviateDataObject(
                # UUID is required by Weaviate; we generate a fresh one for test data.
                uuid=str(uuid4()),
                # Store the raw text in properties for easy inspection later.
                properties={"text": text},
                # Vector payload uses the embedding directly.
                vector=vector,
            )
        ]

    from core.models import VectorPoint

    return [
        VectorPoint(
            # Qdrant allows string IDs; we generate a unique one for test data.
            id=str(uuid4()),
            # Vector payload uses the embedding directly.
            vector=vector,
            # Store raw text in payload for easy inspection later.
            payload={"text": text},
        )
    ]


def _mask_secret(value: str | None) -> str:
    """Mask secrets so logs show presence without leaking values."""
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "********"
    return f"{value[:4]}...{value[-4:]}"


def _print_provider_env() -> None:
    """Print provider-related env vars, masking secrets."""
    import os

    logger.info("External provider config from environment:")
    logger.info("  OLLAMA_BASE_URL=%s", os.getenv("OLLAMA_BASE_URL", ""))
    logger.info("  OLLAMA_MODEL=%s", os.getenv("OLLAMA_MODEL", ""))
    logger.info("  VECTOR_DB_PROVIDER=%s", os.getenv("VECTOR_DB_PROVIDER", ""))
    logger.info("  QDRANT_URL=%s", os.getenv("QDRANT_URL", ""))
    logger.info("  QDRANT_API_KEY=%s", _mask_secret(os.getenv("QDRANT_API_KEY")))
    logger.info("  WEAVIATE_URL=%s", os.getenv("WEAVIATE_URL", ""))
    logger.info("  WEAVIATE_GRPC_HOST=%s", os.getenv("WEAVIATE_GRPC_HOST", ""))
    logger.info("  WEAVIATE_API_KEY=%s", _mask_secret(os.getenv("WEAVIATE_API_KEY")))


def _configure_logging() -> None:
    """Configure logging for the smoke test script."""
    logging.basicConfig(level=logging.INFO, format="[smoke-providers] %(message)s")


def _debug(msg: str) -> None:
    """Lightweight debug logger for step-by-step output."""
    logger.info("%s", msg)


def _weaviate_collection_candidates(collection: str) -> tuple[str, ...]:
    """Return candidate class names to delete in Weaviate."""
    if not collection:
        return (collection,)
    normalized = collection[0].upper() + collection[1:]
    if normalized == collection:
        return (collection,)
    return (collection, normalized)


def _delete_weaviate_collection_rest(url: str | None, api_key: str | None, collection: str) -> None:
    """Delete a Weaviate collection via REST, avoiding gRPC init checks."""
    if not url:
        raise RuntimeError("Weaviate URL unavailable for REST cleanup")
    import requests

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base_url = url.rstrip("/")
    last_status = None
    for attempt in range(1, 4):
        for name in _weaviate_collection_candidates(collection):
            resp = requests.delete(
                f"{base_url}/v1/schema/{name}",
                headers=headers,
                timeout=10,
            )
            last_status = resp.status_code
            if resp.status_code in (200, 204, 404):
                return
        if attempt < 3:
            time.sleep(0.5)
    raise RuntimeError(f"Weaviate REST delete failed with status={last_status}")


async def _cleanup_collection(provider_type: str, vector_db: object, collection: str) -> None:
    """Best-effort cleanup for smoke test collections."""
    try:
        if provider_type == "qdrant":
            qdrant_client = vector_db.client
            if qdrant_client is None:
                _debug("Cleanup skipped: Qdrant client unavailable")
                return
            _debug(f"Cleaning up collection={collection}")
            await qdrant_client.delete_collection(collection_name=collection)
            _debug("Cleanup completed")
            return
        if provider_type == "weaviate":
            import os

            _debug(f"Cleaning up collection={collection} via REST")
            _delete_weaviate_collection_rest(
                vector_db.url or os.getenv("WEAVIATE_URL"),
                vector_db.api_key or os.getenv("WEAVIATE_API_KEY"),
                collection,
            )
            _debug("Cleanup completed via REST")
            return
        _debug(f"Cleanup skipped: unsupported provider_type={provider_type}")
    except Exception as exc:
        _debug(f"Cleanup failed for collection={collection}: {exc}")


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
        elif provider_type == "weaviate":
            weaviate_client = vector_db.client
            if weaviate_client is None:
                break
            async with weaviate_client:
                if await weaviate_client.collections.exists(collection):
                    return
        else:
            break
        await asyncio.sleep(interval_s)
    raise RuntimeError(f"Collection not ready after {timeout_s:.1f}s: {collection}")


async def _run_smoke_test(
    provider_type: str,
    collection: str,
    text: str,
    cleanup: bool,
) -> int:
    """Execute the smoke test steps in an async flow."""
    from clients.interfaces.embedding import create_embedding_provider
    from clients.interfaces.vector_db import create_vector_db_provider
    from config import VectorDBConfig, get_settings

    # Load settings from environment (e.g., .env.local or compose env).
    settings = get_settings()
    _print_provider_env()
    _debug(f"Using embedding provider: {settings.embedding.provider_type}")
    # Build the embedding provider using configured settings.
    embedder = create_embedding_provider(settings.embedding)

    # Generate the embedding vector for the supplied text.
    _debug("Requesting embedding from provider")
    vector = embedder.embed(text)
    vector_size = len(vector)
    if not vector_size:
        raise RuntimeError("Embedding provider returned an empty vector.")
    _debug(f"Embedding returned with vector_size={vector_size}")

    # Restrict to providers supported by the compose/dev setup.
    if provider_type not in ("qdrant", "weaviate"):
        raise RuntimeError(f"Unsupported provider type: {provider_type}")

    # Use settings' vector DB config when it matches the requested provider.
    # Otherwise build a provider-specific config from environment.
    if provider_type == settings.vector_db.provider_type:
        vector_cfg = settings.vector_db
    else:
        vector_cfg = VectorDBConfig.from_env_for_provider(
            provider_type=provider_type,
            namespace=settings.k8s_namespace,
        )

    _debug(f"Using vector DB provider: {provider_type}")
    if provider_type == "qdrant":
        _debug(f"Qdrant URL: {vector_cfg.qdrant_url}")
    if provider_type == "weaviate":
        _debug(f"Weaviate URL: {vector_cfg.weaviate_url}")
        _debug(f"Weaviate gRPC host: {vector_cfg.weaviate_grpc_host or ''}")
    # Instantiate the vector DB provider.
    if provider_type == "weaviate":
        from clients.weaviate import WeaviateClientWrapper

        if vector_cfg.weaviate_url is None:
            raise RuntimeError("Weaviate URL is not configured.")
        _debug("Weaviate skip_init_checks enabled")
        vector_db = WeaviateClientWrapper(
            url=vector_cfg.weaviate_url,
            api_key=vector_cfg.weaviate_api_key,
            grpc_host=vector_cfg.weaviate_grpc_host,
            skip_init_checks=True,
        )
    else:
        vector_db = create_vector_db_provider(vector_cfg)
    # Build provider-specific points for upsert.
    points = _build_points(provider_type, text, vector)

    try:
        # Upsert the test point (ensures the collection exists as needed).
        _debug(f"Upserting 1 point into collection={collection}")
        await vector_db.batch_upsert_async(
            collection=collection,
            points=points,
            vector_size=vector_size,
        )
        _debug("Upsert completed")
        _debug(f"Waiting for collection={collection} to be ready")
        await _wait_for_collection(provider_type, vector_db, collection)
        _debug("Collection is ready")
        # Search with the same vector to confirm the DB returns results.
        _debug(f"Searching collection={collection} with limit=1")
        results = await vector_db.search_async(
            collection=collection,
            query_vector=vector,
            limit=1,
        )
        _debug(f"Search completed with total={results.total}")

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
    # Ensure repo modules can be imported without installation.
    _add_repo_src_to_path()
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
        from config import get_settings

        # Default to the configured provider when no override is given.
        provider_type = get_settings().vector_db.provider_type

    # Run the async smoke test and return its exit code.
    return asyncio.run(_run_smoke_test(provider_type, collection, text, cleanup))


if __name__ == "__main__":
    # Ensure a proper exit code is returned to the shell.
    raise SystemExit(main())

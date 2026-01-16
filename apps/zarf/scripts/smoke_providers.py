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
import sys
from pathlib import Path
from uuid import uuid4


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
        help="Collection name (defaults to smoke_<random>).",
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

    print("External provider config from environment:")
    print(f"  OLLAMA_BASE_URL={os.getenv('OLLAMA_BASE_URL', '')}")
    print(f"  OLLAMA_MODEL={os.getenv('OLLAMA_MODEL', '')}")
    print(f"  VECTOR_DB_PROVIDER={os.getenv('VECTOR_DB_PROVIDER', '')}")
    print(f"  QDRANT_URL={os.getenv('QDRANT_URL', '')}")
    print(f"  QDRANT_API_KEY={_mask_secret(os.getenv('QDRANT_API_KEY'))}")
    print(f"  WEAVIATE_URL={os.getenv('WEAVIATE_URL', '')}")
    print(f"  WEAVIATE_GRPC_HOST={os.getenv('WEAVIATE_GRPC_HOST', '')}")
    print(f"  WEAVIATE_API_KEY={_mask_secret(os.getenv('WEAVIATE_API_KEY'))}")


def _debug(msg: str) -> None:
    """Lightweight debug logger for step-by-step output."""
    print(f"[smoke] {msg}")


async def _run_smoke_test(provider_type: str, collection: str, text: str) -> int:
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
    vector_db = create_vector_db_provider(vector_cfg)
    # Build provider-specific points for upsert.
    points = _build_points(provider_type, text, vector)

    # Upsert the test point (ensures the collection exists as needed).
    _debug(f"Upserting 1 point into collection={collection}")
    await vector_db.batch_upsert_async(
        collection=collection,
        points=points,
        vector_size=vector_size,
    )
    _debug("Upsert completed")
    # Search with the same vector to confirm the DB returns results.
    _debug(f"Searching collection={collection} with limit=1")
    results = await vector_db.search_async(
        collection=collection,
        query_vector=vector,
        limit=1,
    )
    _debug(f"Search completed with total={results.total}")

    # Print a compact summary for humans and CI logs.
    print(
        "provider:",
        provider_type,
        "collection:",
        collection,
        "vector_size:",
        vector_size,
        "results:",
        results.total,
    )
    return 0


def main() -> int:
    """Entry point for CLI execution."""
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
    # Text to embed and store.
    text = args.text

    if provider_type is None:
        from config import get_settings

        # Default to the configured provider when no override is given.
        provider_type = get_settings().vector_db.provider_type

    # Run the async smoke test and return its exit code.
    return asyncio.run(_run_smoke_test(provider_type, collection, text))


if __name__ == "__main__":
    # Ensure a proper exit code is returned to the shell.
    raise SystemExit(main())

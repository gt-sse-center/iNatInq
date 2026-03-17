#!/usr/bin/env python3
"""Generate a filtered INQUIRE dataset containing only queries with full coverage.

Reads the full INQUIRE benchmark JSON, checks which relevant document IDs
actually exist in the Qdrant collection (via UUID5 lookup), and outputs a
filtered dataset containing only queries where ALL relevant documents are
indexed.

Usage:
    python scripts/generate_inquire_subset.py \
        --input benchmarks/inquire/inquire-val.json \
        --output benchmarks/inquire/inquire-val-subset.json \
        --qdrant-url $QDRANT_URL \
        --qdrant-api-key $QDRANT_API_KEY \
        --collection demo-data-1M_images
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _collect_unique_doc_ids(dataset: dict) -> set[str]:
    """Collect all unique relevant document IDs across all queries.

    Args:
        dataset: Parsed dataset dict with ``queries`` list.

    Returns:
        Set of unique document IDs referenced by any query.
    """
    doc_ids: set[str] = set()
    for query in dataset["queries"]:
        doc_ids.update(query["relevant"])
    return doc_ids


def _doc_id_to_uuid5(doc_id: str, s3_prefix: str = "") -> str:
    """Convert a document ID to its UUID5 Qdrant point ID.

    Args:
        doc_id: Plain numeric photo ID.
        s3_prefix: S3 prefix prepended to doc_id when computing UUID5
            (e.g., ``"benchmark-images/"``). Must match the prefix used
            during ingestion so the UUID5 matches the stored point ID.

    Returns:
        UUID5 string.
    """
    return str(uuid5(NAMESPACE_URL, s3_prefix + doc_id))


async def _check_ids_in_qdrant(
    point_ids: list[str],
    *,
    qdrant_url: str,
    qdrant_api_key: str,
    collection: str,
    batch_size: int = 100,
) -> set[str]:
    """Check which UUID5 point IDs exist in the Qdrant collection.

    Uses ``AsyncQdrantClient.retrieve()`` in batches.

    Args:
        point_ids: List of UUID5 strings to check.
        qdrant_url: Qdrant instance URL.
        qdrant_api_key: Qdrant API key.
        collection: Collection name to query.
        batch_size: IDs per retrieve call.

    Returns:
        Set of point IDs that were found in the collection.
    """
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    found: set[str] = set()

    try:
        total = len(point_ids)
        for i in range(0, total, batch_size):
            batch = point_ids[i : i + batch_size]
            logger.info("Checking batch %d-%d of %d", i, min(i + batch_size, total), total)
            points = await client.retrieve(
                collection_name=collection,
                ids=batch,
                with_payload=False,
                with_vectors=False,
            )
            for point in points:
                found.add(str(point.id))
    finally:
        await client.close()

    return found


def _filter_queries(
    dataset: dict,
    present_doc_ids: set[str],
) -> list[dict]:
    """Trim each query's relevant list to only docs present in the collection.

    For each query, intersects its ``relevant`` list with
    ``present_doc_ids``. Queries with no remaining relevant docs are
    dropped entirely.

    Args:
        dataset: Parsed dataset dict.
        present_doc_ids: Set of document IDs confirmed present in Qdrant.

    Returns:
        List of query dicts with trimmed relevant lists.
    """
    filtered = []
    for query in dataset["queries"]:
        trimmed = [doc_id for doc_id in query["relevant"] if doc_id in present_doc_ids]
        if not trimmed:
            continue
        filtered.append({**query, "relevant": trimmed})
    return filtered


def _build_output_dataset(
    source_dataset: dict,
    filtered_queries: list[dict],
    *,
    collection: str,
    indexed_count: int,
    total_doc_ids: int,
    present_count: int,
) -> dict:
    """Build the output dataset dict with provenance metadata.

    Args:
        source_dataset: Original input dataset.
        filtered_queries: Filtered query list.
        collection: Qdrant collection used for filtering.
        indexed_count: Number of indexed doc IDs found.
        total_doc_ids: Total unique doc IDs in the original dataset.
        present_count: Number of doc IDs present in collection.

    Returns:
        Output dataset dict with metadata.
    """
    return {
        "name": f"{source_dataset['name']}-subset",
        "description": (
            f"Filtered subset of {source_dataset['name']} — "
            f"only queries with 100% relevant-doc coverage in collection."
        ),
        "modality": source_dataset["modality"],
        "queries": filtered_queries,
        "metadata": {
            **(source_dataset.get("metadata") or {}),
            "source_dataset": source_dataset["name"],
            "qdrant_collection": collection,
            "indexed_count": indexed_count,
            "total_doc_ids": total_doc_ids,
            "present_doc_ids": present_count,
            "query_coverage": (f"{len(filtered_queries)}/{len(source_dataset['queries'])}"),
        },
    }


async def _run(args: argparse.Namespace) -> None:
    """Main async logic for subset generation.

    Args:
        args: Parsed CLI arguments.
    """
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    # Load and validate input dataset (sync I/O before async work begins)
    logger.info("Loading dataset from %s", input_path)
    raw_text = input_path.read_text(encoding="utf-8")  # noqa: ASYNC240
    raw = json.loads(raw_text)

    if "queries" not in raw or not raw["queries"]:
        logger.error("Input dataset has no queries")
        return

    # Collect all unique doc IDs
    all_doc_ids = _collect_unique_doc_ids(raw)
    logger.info("Found %d unique doc IDs across %d queries", len(all_doc_ids), len(raw["queries"]))

    # Convert to UUID5 point IDs
    s3_prefix = args.s3_prefix or ""
    doc_id_to_uuid: dict[str, str] = {doc_id: _doc_id_to_uuid5(doc_id, s3_prefix) for doc_id in all_doc_ids}
    uuid_to_doc_id: dict[str, str] = {v: k for k, v in doc_id_to_uuid.items()}
    point_ids = list(doc_id_to_uuid.values())
    logger.info("Generated %d UUID5 point IDs for lookup", len(point_ids))

    # Look up which point IDs exist in Qdrant
    found_uuids = await _check_ids_in_qdrant(
        point_ids,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection=args.collection,
        batch_size=args.batch_size,
    )
    present_doc_ids = {uuid_to_doc_id[uid] for uid in found_uuids if uid in uuid_to_doc_id}
    logger.info(
        "Found %d/%d doc IDs present in collection '%s'",
        len(present_doc_ids),
        len(all_doc_ids),
        args.collection,
    )

    # Filter queries
    filtered_queries = _filter_queries(raw, present_doc_ids)
    logger.info(
        "Filtered: %d/%d queries have full coverage",
        len(filtered_queries),
        len(raw["queries"]),
    )

    # Build output
    output = _build_output_dataset(
        raw,
        filtered_queries,
        collection=args.collection,
        indexed_count=len(found_uuids),
        total_doc_ids=len(all_doc_ids),
        present_count=len(present_doc_ids),
    )

    if args.dry_run:
        logger.info("Dry run — not writing output file")
        logger.info("Output would contain %d queries", len(filtered_queries))
        return

    if output_path is None:
        logger.error("--output is required when not in --dry-run mode")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s (%d queries)", output_path, len(filtered_queries))


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for generating INQUIRE subset datasets."""
    parser = argparse.ArgumentParser(
        description="Generate a filtered INQUIRE dataset with full collection coverage."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the full INQUIRE JSON dataset.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path for the filtered output JSON.",
    )
    parser.add_argument(
        "--qdrant-url",
        required=True,
        help="Remote Qdrant instance URL.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        required=True,
        help="Qdrant API key.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name to check for coverage.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="S3 key prefix prepended to doc IDs during ingestion (e.g., 'benchmark-images/').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of IDs per retrieve call (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without writing output file.",
    )

    args = parser.parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert INQUIRE benchmark CSVs to gold-standard JSON files.

Downloads the three CSV files from the INQUIRE GitHub repository and produces
two JSON dataset files (val + test) matching our gold-standard.schema.json.

Usage:
    python scripts/convert_inquire.py --output-dir benchmarks/inquire/
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_BASE_URL = "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/main/data/inquire"
_QUERIES_VAL_URL = f"{_BASE_URL}/inquire_queries_val.csv"
_QUERIES_TEST_URL = f"{_BASE_URL}/inquire_queries_test.csv"
_ANNOTATIONS_URL = f"{_BASE_URL}/inquire_annotations.csv"


def _download_csv(url: str) -> str:
    """Download a CSV file and return its text content."""
    logger.info("Downloading %s", url)
    with urlopen(url) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _parse_queries(text: str) -> dict[str, str]:
    """Parse a queries CSV into {query_id: query_text}.

    The CSV has an unnamed index column, then query_id, query_text, and
    metadata columns we discard.
    """
    reader = csv.DictReader(io.StringIO(text))
    queries: dict[str, str] = {}
    for row in reader:
        qid = row["query_id"]
        queries[qid] = row["query_text"]
    return queries


def _parse_annotations(text: str) -> dict[str, list[str]]:
    """Parse annotations CSV into {query_id: [image_id, ...]}."""
    reader = csv.DictReader(io.StringIO(text))
    annotations: dict[str, list[str]] = defaultdict(list)
    for row in reader:
        annotations[row["query_id"]].append(row["image_id"])
    return dict(annotations)


def _build_dataset(
    *,
    name: str,
    description: str,
    queries: dict[str, str],
    annotations: dict[str, list[str]],
) -> dict:
    """Build a gold-standard dataset dict from queries and annotations."""
    query_list = []
    for qid in sorted(queries, key=int):
        relevant = annotations.get(qid, [])
        if not relevant:
            logger.warning("Query %s (%s) has no annotations — skipping", qid, queries[qid])
            continue
        query_list.append(
            {
                "id": qid,
                "text": queries[qid],
                "relevant": [str(img_id) for img_id in relevant],
            }
        )

    return {
        "name": name,
        "description": description,
        "modality": "image",
        "queries": query_list,
        "metadata": {
            "created_at": "2024-10-08",
            "annotators": ["inquire-benchmark"],
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert INQUIRE CSVs to gold-standard JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/inquire"),
        help="Directory for output JSON files (default: benchmarks/inquire/)",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download all three CSVs
    val_text = _download_csv(_QUERIES_VAL_URL)
    test_text = _download_csv(_QUERIES_TEST_URL)
    ann_text = _download_csv(_ANNOTATIONS_URL)

    # Parse
    val_queries = _parse_queries(val_text)
    test_queries = _parse_queries(test_text)
    annotations = _parse_annotations(ann_text)

    logger.info(
        "Parsed %d val queries, %d test queries, %d annotated query IDs",
        len(val_queries),
        len(test_queries),
        len(annotations),
    )

    # Build datasets
    val_dataset = _build_dataset(
        name="inquire-val",
        description="INQUIRE validation split — 50 expert text-to-image queries over iNat24.",
        queries=val_queries,
        annotations=annotations,
    )
    test_dataset = _build_dataset(
        name="inquire-test",
        description="INQUIRE test split — 200 expert text-to-image queries over iNat24.",
        queries=test_queries,
        annotations=annotations,
    )

    # Write
    val_path = output_dir / "inquire-val.json"
    test_path = output_dir / "inquire-test.json"

    for path, dataset in [(val_path, val_dataset), (test_path, test_dataset)]:
        path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Wrote %s (%d queries)", path, len(dataset["queries"]))

    logger.info("Done.")


if __name__ == "__main__":
    main()

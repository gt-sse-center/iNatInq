"""Streaming S3 pagination → filter → batch utilities for image pipelines."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.ingestion.interfaces.operations import ImageContentFetcher

if TYPE_CHECKING:
    from collections.abc import Generator

    from clients.s3 import S3ClientWrapper

logger = logging.getLogger("pipeline.batching")


def iter_image_batches(
    s3: S3ClientWrapper,
    bucket: str,
    prefix: str,
    processed: set[str],
    batch_size: int,
    max_items: int | None,
    page_size: int = 1000,
) -> Generator[list[str], None, None]:
    """Stream S3 pages, filter for images, skip checkpointed keys, and yield batches.

    Consumes ``s3.iter_objects`` lazily so that Ray/Databricks tasks can be
    submitted while S3 pagination is still in progress.

    Args:
        s3: S3 client wrapper with ``iter_objects`` support.
        bucket: S3 bucket name.
        prefix: S3 key prefix to list under.
        processed: Set of already-processed keys (from checkpoint). Keys in
            this set are skipped.
        batch_size: Number of keys per yielded batch.
        max_items: Optional cap on total keys yielded. ``None`` means no limit.
        page_size: Maximum keys per S3 API page (default: 1000).

    Yields:
        Lists of S3 keys, each up to *batch_size* long.
    """
    buffer: list[str] = []
    total_yielded = 0
    pages_consumed = 0
    batches_yielded = 0
    skipped = 0

    for page_keys in s3.iter_objects(bucket=bucket, prefix=prefix, page_size=page_size):
        pages_consumed += 1
        filtered = ImageContentFetcher.filter_image_keys(page_keys)
        page_skipped = 0
        for key in filtered:
            if key in processed:
                skipped += 1
                page_skipped += 1
                continue
            buffer.append(key)
            if len(buffer) >= batch_size:
                batches_yielded += 1
                total_yielded += len(buffer)
                logger.info(
                    "iter_image_batches: yielding batch %d (%d keys, %d total yielded)",
                    batches_yielded,
                    len(buffer),
                    total_yielded,
                    extra={
                        "batch_num": batches_yielded,
                        "batch_size": len(buffer),
                        "total_yielded": total_yielded,
                        "pages_consumed": pages_consumed,
                    },
                )
                yield buffer
                buffer = []
                if max_items is not None and total_yielded >= max_items:
                    logger.info(
                        "iter_image_batches: max_items reached (%d), stopping",
                        max_items,
                    )
                    return
        logger.info(
            "iter_image_batches: page %d — %d raw, %d filtered, %d checkpoint-skipped",
            pages_consumed,
            len(page_keys),
            len(filtered),
            page_skipped,
            extra={
                "page": pages_consumed,
                "raw_keys": len(page_keys),
                "filtered_keys": len(filtered),
                "skipped": page_skipped,
            },
        )

    if buffer:
        batches_yielded += 1
        total_yielded += len(buffer)
        logger.info(
            "iter_image_batches: yielding final batch %d (%d keys)",
            batches_yielded,
            len(buffer),
            extra={"batch_num": batches_yielded, "batch_size": len(buffer)},
        )
        yield buffer

    logger.info(
        "iter_image_batches: done — %d pages, %d batches, %d keys yielded, %d skipped",
        pages_consumed,
        batches_yielded,
        total_yielded,
        skipped,
        extra={
            "total_pages": pages_consumed,
            "total_batches": batches_yielded,
            "total_yielded": total_yielded,
            "total_skipped": skipped,
        },
    )

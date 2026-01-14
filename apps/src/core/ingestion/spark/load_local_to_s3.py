"""Spark job to load local files to S3/MinIO (driver-side batch processing).

This job:
1. Discovers local files on the driver (PVC-mounted)
2. Reads files in chunks (10k at a time) to manage memory
3. Distributes (filename, content) tuples to executors
4. Executors upload directly to S3/MinIO in parallel

This approach avoids loading all files into memory at once by processing
in batches. The driver reads files from the PVC and distributes content
to executors for parallel upload.

REQUIRES:
- PVC mounted on driver pod at /data/testdata
- Spark running in local or cluster mode
"""

import logging
import os
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable
from concurrent import futures
from logging.config import dictConfig
from pathlib import Path

import attrs

from clients.s3 import S3ClientWrapper
from config import MinIOConfig, SparkJobConfig
from foundation.logger import LOGGING_CONFIG

from .spark_config import create_spark_session

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.spark.ingest")


# =============================================================================
# Type Definitions
# =============================================================================


@attrs.define(frozen=True, slots=True)
class ChunkResult:
    """Result of processing a single chunk."""

    chunk_num: int
    success_count: int
    failure_count: int
    read_seconds: float
    upload_seconds: float


@attrs.define(frozen=True, slots=True)
class IngestionResult:
    """Final result of the ingestion job."""

    total_files: int
    successful: int
    failed: int
    elapsed_seconds: float

    @property
    def files_per_second(self) -> float:
        """Calculate upload rate."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return round(self.successful / self.elapsed_seconds, 2)


# =============================================================================
# Code Distribution
# =============================================================================


def distribute_pipeline_code(sc) -> None:
    """Distribute pipeline code to Spark executors.

    Args:
        sc: Spark context.
    """
    src_root = Path("/app/src")
    pipeline_src = src_root / "pipeline"

    if not pipeline_src.exists():
        logger.warning("Pipeline source not found, skipping distribution")
        return

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for py_file in pipeline_src.rglob("*.py"):
            arcname = str(py_file.relative_to(src_root))
            zipf.write(py_file, arcname)

    sc.addPyFile(str(zip_path))
    logger.info("Distributed pipeline package to executors", extra={"zip": str(zip_path)})


# =============================================================================
# File Discovery
# =============================================================================


def discover_files(local_dir: str) -> list[str]:
    """Discover all .txt files in a directory, filtering macOS metadata.

    Args:
        local_dir: Path to local directory.

    Returns:
        List of file paths.

    Raises:
        SystemExit: If directory doesn't exist.
    """
    base_path = Path(local_dir)

    if not base_path.exists():
        logger.error("Local directory not found", extra={"path": local_dir})
        sys.exit(1)

    # Find all .txt files, but filter out macOS metadata files (._*)
    all_files = list(base_path.glob("*.txt"))
    files = [str(p) for p in all_files if not p.name.startswith("._") and not p.name.endswith(".DS_Store")]

    # Log if we filtered out any metadata files
    filtered_count = len(all_files) - len(files)
    if filtered_count > 0:
        logger.info(
            "Filtered out macOS metadata files",
            extra={"filtered": filtered_count, "total_found": len(all_files)},
        )

    if files:
        logger.info("Discovered files", extra={"count": len(files)})

    return files


# =============================================================================
# Chunk Reading
# =============================================================================


def read_chunk_files(
    chunk_files: list[str],
    chunk_num: int,
    chunk_start_time: float,
) -> list[tuple[str, bytes]]:
    """Read a chunk of files into memory.

    Args:
        chunk_files: List of file paths to read.
        chunk_num: Chunk number for logging.
        chunk_start_time: Start time for progress calculations.

    Returns:
        List of (filename, content) tuples.
    """
    chunk_data: list[tuple[str, bytes]] = []
    read_progress_interval = max(10000, len(chunk_files) // 10)

    for idx, file_path in enumerate(chunk_files):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            chunk_data.append((Path(file_path).name, content))

            # Progress logging during file reading
            should_log = (idx + 1) % read_progress_interval == 0 or (idx + 1) == len(chunk_files)
            if should_log:
                elapsed = time.time() - chunk_start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Reading chunk files",
                    extra={
                        "chunk": chunk_num,
                        "files_read": idx + 1,
                        "total_in_chunk": len(chunk_files),
                        "elapsed_seconds": round(elapsed, 2),
                        "files_per_second": round(rate, 2),
                    },
                )
        except Exception as e:
            logger.exception(
                "Failed to read file",
                extra={"file": file_path, "error": str(e)},
            )

    return chunk_data


# =============================================================================
# Executor-Side Upload
# =============================================================================


def create_upload_partition_fn(
    *,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str,
    prefix: str,
) -> Callable[[Iterable[tuple[str, bytes]]], Iterable[tuple[str, int]]]:
    """Create a mapPartitions function that runs on executors.

    Executors:
    - Receive (filename, content) tuples from driver
    - Upload directly to S3/MinIO

    Args:
        endpoint_url: S3 endpoint URL.
        access_key_id: S3 access key.
        secret_access_key: S3 secret key.
        bucket: S3 bucket name.
        prefix: S3 key prefix.

    Returns:
        Function suitable for mapPartitions.
    """

    def upload_partition(
        file_data: Iterable[tuple[str, bytes]],
    ) -> Iterable[tuple[str, int]]:
        """Upload files from (filename, content) tuples."""
        s3 = S3ClientWrapper(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

        file_data_list = list(file_data)
        if not file_data_list:
            return

        def process(filename_content: tuple[str, bytes]) -> tuple[str, int]:
            filename, content = filename_content
            key = f"{prefix.rstrip('/')}/{filename}"

            try:
                # Idempotency check
                if s3.exists(bucket=bucket, key=key):
                    return ("success", 1)

                s3.put_object(bucket=bucket, key=key, body=content)
                return ("success", 1)

            except Exception as e:
                logger.exception(
                    "Upload failed",
                    extra={"file": filename, "error": str(e)},
                )
                return ("failure", 1)

        # Threaded uploads per executor
        max_workers = min(len(file_data_list), 16)

        with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            yield from pool.map(process, file_data_list)

    return upload_partition


# =============================================================================
# Chunk Processing
# =============================================================================


def calculate_partitions(chunk_size: int, default_parallelism: int) -> int:
    """Calculate optimal partition count for a chunk.

    Args:
        chunk_size: Number of files in chunk.
        default_parallelism: Spark default parallelism.

    Returns:
        Number of partitions.
    """
    files_per_partition = 5000
    calculated = (chunk_size + files_per_partition - 1) // files_per_partition
    partitions = min(calculated, 1000)
    return max(partitions, default_parallelism)


def upload_chunk(
    sc,
    chunk_data: list[tuple[str, bytes]],
    upload_fn: Callable[[Iterable[tuple[str, bytes]]], Iterable[tuple[str, int]]],
    chunk_num: int,
    default_parallelism: int,
) -> tuple[int, int, float]:
    """Upload a chunk of files via Spark RDD.

    Args:
        sc: Spark context.
        chunk_data: List of (filename, content) tuples.
        upload_fn: Upload partition function.
        chunk_num: Chunk number for logging.
        default_parallelism: Spark default parallelism.

    Returns:
        Tuple of (success_count, failure_count, upload_seconds).
    """
    partitions = calculate_partitions(len(chunk_data), default_parallelism)

    logger.info(
        "Starting chunk upload",
        extra={
            "chunk": chunk_num,
            "partitions": partitions,
            "files_to_upload": len(chunk_data),
        },
    )

    upload_start = time.time()

    results = (
        sc.parallelize(chunk_data, partitions)
        .mapPartitions(upload_fn)
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    )

    upload_elapsed = time.time() - upload_start
    success = results.get("success", 0)
    failure = results.get("failure", 0)

    return success, failure, upload_elapsed


def process_chunk(
    sc,
    files: list[str],
    chunk_start_idx: int,
    chunk_size: int,
    upload_fn: Callable[[Iterable[tuple[str, bytes]]], Iterable[tuple[str, int]]],
    total_files: int,
    total_chunks: int,
    default_parallelism: int,
    running_totals: tuple[int, int],
) -> ChunkResult:
    """Process a single chunk: read files and upload.

    Args:
        sc: Spark context.
        files: All file paths.
        chunk_start_idx: Starting index for this chunk.
        chunk_size: Number of files per chunk.
        upload_fn: Upload partition function.
        total_files: Total files being processed.
        total_chunks: Total number of chunks.
        default_parallelism: Spark default parallelism.
        running_totals: Tuple of (total_success, total_failure) so far.

    Returns:
        ChunkResult with processing statistics.
    """
    chunk_files = files[chunk_start_idx : chunk_start_idx + chunk_size]
    chunk_num = (chunk_start_idx // chunk_size) + 1
    chunk_start_time = time.time()

    logger.info(
        "Processing chunk",
        extra={
            "chunk": chunk_num,
            "total_chunks": total_chunks,
            "chunk_size": len(chunk_files),
            "files_processed": chunk_start_idx,
            "progress_pct": round((chunk_start_idx / total_files) * 100, 2),
        },
    )

    # Read files
    chunk_data = read_chunk_files(chunk_files, chunk_num, chunk_start_time)
    read_elapsed = time.time() - chunk_start_time

    if not chunk_data:
        logger.warning("No files were successfully read in chunk", extra={"chunk": chunk_num})
        return ChunkResult(
            chunk_num=chunk_num,
            success_count=0,
            failure_count=0,
            read_seconds=read_elapsed,
            upload_seconds=0.0,
        )

    logger.info(
        "Chunk files read",
        extra={
            "chunk": chunk_num,
            "files_read": len(chunk_data),
            "elapsed_seconds": round(read_elapsed, 2),
            "files_per_second": round(len(chunk_data) / read_elapsed, 2) if read_elapsed > 0 else 0,
        },
    )

    # Upload files
    success, failure, upload_elapsed = upload_chunk(sc, chunk_data, upload_fn, chunk_num, default_parallelism)

    # Calculate running totals
    new_total_success = running_totals[0] + success
    progress_pct = round((new_total_success / total_files) * 100, 2)

    logger.info(
        "Chunk complete",
        extra={
            "chunk": chunk_num,
            "successful": success,
            "failed": failure,
            "total_successful": new_total_success,
            "total_failed": running_totals[1] + failure,
            "read_seconds": round(read_elapsed, 2),
            "upload_seconds": round(upload_elapsed, 2),
            "total_seconds": round(time.time() - chunk_start_time, 2),
            "upload_files_per_second": round(success / upload_elapsed, 2) if upload_elapsed > 0 else 0,
            "overall_progress_pct": progress_pct,
        },
    )

    return ChunkResult(
        chunk_num=chunk_num,
        success_count=success,
        failure_count=failure,
        read_seconds=read_elapsed,
        upload_seconds=upload_elapsed,
    )


# =============================================================================
# Main Orchestration
# =============================================================================


def run_ingestion(
    spark,
    files: list[str],
    minio: MinIOConfig,
    s3_prefix: str,
    chunk_size: int = 10000,
) -> IngestionResult:
    """Run the complete ingestion pipeline.

    Args:
        spark: Spark session.
        files: List of file paths to process.
        minio: MinIO configuration.
        s3_prefix: S3 key prefix.
        chunk_size: Files per chunk (default: 10000).

    Returns:
        IngestionResult with final statistics.
    """
    start = time.time()
    sc = spark.sparkContext
    total_files = len(files)
    total_chunks = (total_files + chunk_size - 1) // chunk_size

    upload_fn = create_upload_partition_fn(
        endpoint_url=minio.endpoint_url,
        access_key_id=minio.access_key_id,
        secret_access_key=minio.secret_access_key,
        bucket=minio.bucket,
        prefix=s3_prefix,
    )

    total_success = 0
    total_failure = 0

    for chunk_start in range(0, len(files), chunk_size):
        result = process_chunk(
            sc=sc,
            files=files,
            chunk_start_idx=chunk_start,
            chunk_size=chunk_size,
            upload_fn=upload_fn,
            total_files=total_files,
            total_chunks=total_chunks,
            default_parallelism=sc.defaultParallelism,
            running_totals=(total_success, total_failure),
        )

        total_success += result.success_count
        total_failure += result.failure_count

    elapsed = time.time() - start

    return IngestionResult(
        total_files=total_files,
        successful=total_success,
        failed=total_failure,
        elapsed_seconds=round(elapsed, 2),
    )


def main() -> None:
    """Load local files from PVC to S3/MinIO using Spark driver-side batch processing."""
    # Configuration
    local_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LOCAL_DIR", "/data/testdata/inputs")
    s3_prefix = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("S3_PREFIX", "inputs/")
    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    pod_ip = os.environ.get("POD_IP", "localhost")

    minio = MinIOConfig.from_env(namespace=namespace)
    spark_cfg = SparkJobConfig.from_env(namespace=namespace)

    spark = create_spark_session(
        pod_ip=pod_ip,
        spark_config=spark_cfg,
        app_name="LoadLocalToS3",
        s3_endpoint=minio.endpoint_url,
        s3_access_key=minio.access_key_id,
        s3_secret_key=minio.secret_access_key,
    )

    # Distribute pipeline code to executors
    distribute_pipeline_code(spark.sparkContext)

    # Discover files
    files = discover_files(local_dir)

    if not files:
        logger.warning("No files found", extra={"path": local_dir})
        spark.stop()
        return

    # Run ingestion
    result = run_ingestion(spark, files, minio, s3_prefix)

    logger.info(
        "Ingestion complete",
        extra={
            "total_files": result.total_files,
            "successful": result.successful,
            "failed": result.failed,
            "elapsed_seconds": result.elapsed_seconds,
            "files_per_second": result.files_per_second,
        },
    )

    spark.stop()

    if result.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

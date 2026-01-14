"""Spark job: S3 → Ollama embeddings → Qdrant/Weaviate.

Driver responsibilities:
- Validate configuration
- Build RDD
- Launch executor-side async processing

Note: Code distribution to executors is handled by Spark Operator via deps.pyFiles.
"""

import logging
import os
import sys
import time
from logging.config import dictConfig
from pathlib import Path

from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from config import EmbeddingConfig, MinIOConfig, SparkJobConfig, VectorDBConfig
from core.ingestion.checkpoint import CheckpointManager, is_s3_path
from core.ingestion.spark.processing import process_partition_async_wrapper
from core.ingestion.spark.spark_config import create_spark_session
from foundation.logger import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.spark")


def main() -> None:
    """Process S3 objects and store embeddings in Qdrant using Spark.

    Orchestrates the complete pipeline:
    1. Loads configuration from environment variables
    2. Creates and configures Spark session
    3. Lists S3 objects matching the prefix
    4. Loads checkpoint (if enabled) to skip already processed objects
    5. Processes objects in parallel using Spark RDD partitions
    6. Saves checkpoint (if enabled) with successfully processed keys
    7. Logs job statistics and stops Spark session

    Note: Code distribution to executors is handled by Spark Operator via deps.pyFiles
    (pipeline.zip). No manual code distribution is needed.

    Configuration is loaded from environment variables:
    - K8S_NAMESPACE: Kubernetes namespace (default: "ml-system")
    - S3_PREFIX: S3 prefix to process (default: "inputs/")
    - Additional config via SparkJobConfig, MinIOConfig, VectorDBConfig, EmbeddingConfig

    The job processes S3 objects by:
    - Fetching object content from S3
    - Generating embeddings via Ollama
    - Upserting vectors to Qdrant collection

    Raises:
        SystemExit: If S3 listing fails or other critical errors occur.
    """
    start = time.time()
    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    s3_prefix = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("S3_PREFIX", "inputs/")

    spark_cfg = SparkJobConfig.from_env(namespace)
    minio_cfg = MinIOConfig.from_env(namespace)
    vector_cfg = VectorDBConfig.from_env(namespace)
    embed_cfg = EmbeddingConfig.from_env(namespace)

    spark = create_spark_session(
        pod_ip=os.environ.get("POD_IP", "localhost"),
        spark_config=spark_cfg,
        app_name="s3-to-vector-db",
        s3_endpoint=minio_cfg.endpoint_url,
        s3_access_key=minio_cfg.access_key_id,
        s3_secret_key=minio_cfg.secret_access_key,
    )

    sc = spark.sparkContext
    # Code distribution is handled by Spark Operator via deps.pyFiles (pipeline.zip)

    s3 = S3ClientWrapper(
        endpoint_url=minio_cfg.endpoint_url,
        access_key_id=minio_cfg.access_key_id,
        secret_access_key=minio_cfg.secret_access_key,
    )

    try:
        keys = s3.list_objects(bucket=minio_cfg.bucket, prefix=s3_prefix)
    except ClientError as e:
        logger.exception("Failed to list S3 objects", extra={"error": str(e)})
        spark.stop()
        sys.exit(1)

    checkpoint_manager = CheckpointManager(s3_client=s3 if is_s3_path(spark_cfg.checkpoint_dir) else None)
    if spark_cfg.checkpoint_enabled:
        checkpoint_path = (
            f"{spark_cfg.checkpoint_dir}/{vector_cfg.collection}.json"
            if is_s3_path(spark_cfg.checkpoint_dir)
            else Path(spark_cfg.checkpoint_dir) / f"{vector_cfg.collection}.json"
        )
        processed = checkpoint_manager.load(checkpoint_path)
        keys = [k for k in keys if k not in processed]

    if not keys:
        logger.info("No new objects to process")
        spark.stop()
        return

    partitions = min(
        max(len(keys) // spark_cfg.partition_target_size, sc.defaultParallelism),
        spark_cfg.max_partitions,
    )

    rdd = sc.parallelize(keys, partitions)

    results = rdd.mapPartitions(
        lambda part: process_partition_async_wrapper(
            keys=part,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key=minio_cfg.access_key_id,
            s3_secret_key=minio_cfg.secret_access_key,
            s3_bucket=minio_cfg.bucket,
            embedding_config=embed_cfg,
            collection=vector_cfg.collection,
            ollama_max_concurrency=getattr(spark_cfg, "ollama_max_concurrency", 10),
            ollama_rps=getattr(spark_cfg, "ollama_requests_per_second", 5),
            min_embed_batch=getattr(spark_cfg, "embed_batch_min", 1),
            max_embed_batch=getattr(spark_cfg, "embed_batch_max", 8),
        )
    ).collect()

    success = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - success

    if spark_cfg.checkpoint_enabled:
        processed.update(k for k, ok, _ in results if ok)
        checkpoint_manager.save(checkpoint_path, processed)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "Job complete",
        extra={
            "successful": success,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "rate_per_sec": round(len(results) / elapsed, 2),
        },
    )

    spark.stop()


if __name__ == "__main__":
    main()

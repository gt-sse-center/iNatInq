"""Spark session configuration and setup.

This module provides utilities for creating and configuring Spark sessions
for the pipeline Spark jobs. All configuration is loaded from SparkJobConfig
to make settings easily configurable via environment variables.
"""

import logging
import sys
from typing import Any, cast

from pyspark.sql import SparkSession

from config import SparkJobConfig

logger = logging.getLogger("pipeline.spark")


def create_spark_session(
    pod_ip: str,
    spark_config: SparkJobConfig,
    app_name: str = "PipelineSparkJob",
    s3_endpoint: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
) -> Any:
    """Create and configure a Spark session for pipeline processing.

    Args:
        pod_ip: Pod IP address (injected by Kubernetes).
        spark_config: Spark job configuration (master URL, resources, etc.).
        app_name: Spark application name (default: "PipelineSparkJob").
        s3_endpoint: Optional S3 service endpoint URL (required if using S3A filesystem).
        s3_access_key: Optional S3 access key (required if using S3A filesystem).
        s3_secret_key: Optional S3 secret key (required if using S3A filesystem).

    Returns:
        Configured SparkSession instance.

    Raises:
        ImportError: If PySpark is not installed.

    Example:
        ```python
        import os
        from config import SparkJobConfig
        from core.ingestion.spark.spark_config import create_spark_session

        namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
        config = SparkJobConfig.from_env(namespace=namespace)
        spark = create_spark_session(
            pod_ip="10.244.0.1",
            s3_endpoint="http://minio.ml-system:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            spark_config=config,
        )
        ```

    Note:
        The Spark session is configured with:
        - Cluster mode (connects to external Spark master)
        - Resource configuration (executor memory, cores, driver memory)
        - Proper driver networking for Kubernetes pods
        - S3A filesystem configuration for MinIO/S3 access
        - Optional RDD checkpointing for fault tolerance (supports S3 URIs)
    """
    # Create Spark session builder
    # Note: PySpark's builder pattern uses dynamic attributes that type checkers
    # don't understand. We use cast() to help with type checking.
    builder = cast("Any", SparkSession.builder)
    spark_builder = builder.appName(app_name)

    # Use configured master URL (cluster mode or local mode)
    # For cluster mode: spark://spark-master:7077
    # For local mode: local[*] (useful for testing or when workers unavailable)
    # When using Spark Operator, don't set master URL - Spark Operator manages it
    if spark_config.master_url:
        spark_builder = spark_builder.master(spark_config.master_url)

    # Configure Python paths to ensure consistency
    python_executable = sys.executable
    spark_builder = spark_builder.config("spark.pyspark.python", python_executable)
    spark_builder = spark_builder.config("spark.pyspark.driver.python", python_executable)

    # Note: PYTHONPATH is automatically set by Spark Operator when using deps.pyFiles
    # The pipeline.zip is extracted and added to PYTHONPATH automatically
    # We don't override PYTHONPATH here to allow Spark to manage it

    # Configure executor Java options for logging
    spark_builder = spark_builder.config(
        "spark.executor.extraJavaOptions",
        "-Dlog4j.configuration=file:/opt/spark/conf/log4j.properties",
    )

    # Ensure Python output is unbuffered for better log visibility
    spark_builder = spark_builder.config("spark.executorEnv.PYTHONUNBUFFERED", "1")

    # Resource configuration (only applies in cluster mode)
    spark_builder = spark_builder.config("spark.executor.memory", spark_config.executor_memory)
    spark_builder = spark_builder.config("spark.executor.cores", str(spark_config.executor_cores))
    spark_builder = spark_builder.config("spark.driver.memory", spark_config.driver_memory)
    spark_builder = spark_builder.config(
        "spark.default.parallelism", str(spark_config.default_parallelism)
    )
    spark_builder = spark_builder.config(
        "spark.sql.shuffle.partitions", str(spark_config.shuffle_partitions)
    )

    # Configure driver networking for Kubernetes (required for cluster mode)
    # bindAddress: Listen on all interfaces (0.0.0.0)
    # host: Advertise pod IP/DNS so executors can connect back
    spark_builder = spark_builder.config("spark.driver.bindAddress", "0.0.0.0")
    spark_builder = spark_builder.config("spark.driver.host", pod_ip)
    spark_builder = spark_builder.config("spark.driver.port", "4040")
    spark_builder = spark_builder.config("spark.driver.blockManager.port", "4041")

    logger.debug(
        "Configured Spark driver networking",
        extra={
            "driver_host": pod_ip,
            "driver_bindAddress": "0.0.0.0",
            "driver_port": "4040",
            "blockManager_port": "4041",
        },
    )
    spark_builder = spark_builder.config("spark.network.timeout", spark_config.network_timeout)
    spark_builder = spark_builder.config(
        "spark.executor.heartbeatInterval", spark_config.heartbeat_interval
    )

    # Configure max frame size to handle large result batches
    # Default is 128MB (134217728 bytes), increase to 256MB for large datasets
    # This prevents "Too large frame" errors when transferring results from executors
    spark_builder = spark_builder.config("spark.rpc.message.maxSize", "256")  # In MB

    # Configure S3A filesystem for MinIO/S3 access (only if credentials provided)
    if s3_endpoint and s3_access_key and s3_secret_key:
        spark_builder = spark_builder.config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        spark_builder = spark_builder.config("spark.hadoop.fs.s3a.access.key", s3_access_key)
        spark_builder = spark_builder.config("spark.hadoop.fs.s3a.secret.key", s3_secret_key)
        spark_builder = spark_builder.config("spark.hadoop.fs.s3a.path.style.access", "true")
        spark_builder = spark_builder.config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

    # Configure checkpointing for Spark RDD fault tolerance
    # Supports both local paths and S3 URIs (e.g., s3a://bucket/checkpoints/)
    if spark_config.checkpoint_dir:
        spark_builder = spark_builder.config("spark.checkpoint.dir", spark_config.checkpoint_dir)

    spark = spark_builder.getOrCreate()

    if spark_config.checkpoint_dir:
        logger.info(
            "Spark session created",
            extra={
                "master": spark_config.master_url,
                "executor_memory": spark_config.executor_memory,
                "executor_cores": spark_config.executor_cores,
                "driver_memory": spark_config.driver_memory,
                "parallelism": spark_config.default_parallelism,
                "checkpoint_dir": spark_config.checkpoint_dir,
            },
        )

    return spark


# Spark Benchmarks

This folder provides a Spark-based benchmark driver that mirrors the existing
download -> embed -> build -> search -> update flow, but distributes ingestion
and querying across Spark executors.

## Requirements
- `pyspark` available on the driver and executors
- Shared storage for datasets/embeddings (NFS/S3/ADLS/etc.)

## Config
You can reuse existing YAML configs and optionally add a `spark` block:

```yaml
spark:
  master: "local[*]"
  app_name: "inatinqperf-spark"
  partitions: 200
  force_local_fs: true
  clear_hadoop_conf: true
  use_parquet_embeddings: true
  conf:
    spark.executor.memory: "4g"
    spark.sql.shuffle.partitions: "200"
```

## Usage
```bash
python spark/run_benchmark_spark.py configs/inquire_benchmark_weaviate.yaml
```

Notes:
- The Spark runner reads HuggingFace embeddings from `cfg.embedding.directory` and converts
  them to parquet for Spark to consume. If embeddings are missing, it computes them on the driver.
- The parquet embeddings are stored under `cfg.embedding.directory` as `spark_embeddings.parquet`.
- If parquet loading fails (or `use_parquet_embeddings: false`), the runner reads the HuggingFace
  dataset shards directly on executors.

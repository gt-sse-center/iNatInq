"""Unit tests for core.ingestion.pipeline module.

This module contains comprehensive unit tests for the IngestionPipeline class,
which orchestrates the ingestion of S3 objects into a vector database using Ray
for distributed processing.

# Test Coverage

- JobResult dataclass creation and validation
- IngestionPipeline initialization with explicit configs
- IngestionPipeline.from_env() factory method
- Pipeline run() lifecycle (init, execute, shutdown)
- Error handling and cleanup guarantees
- Batch creation and processing
- Checkpoint loading and saving
- Content type routing (text vs images)

# Running Tests

    pytest tests/unit/core/ingestion/test_pipeline.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from config import EmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig
from core.ingestion.pipeline import IngestionPipeline, JobResult


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_creates_job_result_with_all_fields(self):
        """Verify JobResult is created with all required fields.

        **Why this test is important:**

        - JobResult is the primary return type from pipeline execution
        - All fields must be properly stored for accurate reporting
        - Downstream systems depend on these metrics for monitoring

        **What it tests:**

        - All five fields (successful, failed, total, elapsed_seconds, rate_per_second)
          are correctly assigned
        - Field values are preserved exactly as provided
        """
        result = JobResult(
            successful=90,
            failed=10,
            total=100,
            elapsed_seconds=5.5,
            rate_per_second=18.18,
        )
        assert result.successful == 90
        assert result.failed == 10
        assert result.total == 100
        assert result.elapsed_seconds == 5.5
        assert result.rate_per_second == 18.18

    def test_job_result_allows_zero_values(self):
        """Verify JobResult accepts zero values for all numeric fields.

        **Why this test is important:**

        - Zero values are valid for empty runs or edge cases
        - Ensures no accidental None coercion or validation errors
        - Critical for handling scenarios with no files to process

        **What it tests:**

        - JobResult can be instantiated with all zeros
        - Zero values are stored without transformation
        """
        result = JobResult(
            successful=0,
            failed=0,
            total=0,
            elapsed_seconds=0.0,
            rate_per_second=0.0,
        )
        assert result.total == 0


class TestIngestionPipelineInit:
    """Tests for IngestionPipeline initialization."""

    @pytest.fixture
    def mock_cluster_strategy(self):
        """Create a mock cluster strategy."""
        strategy = MagicMock()
        strategy.init = MagicMock()
        strategy.shutdown = MagicMock()
        strategy.config = MagicMock()
        return strategy

    @pytest.fixture
    def ray_config(self):
        """Create a RayJobConfig for testing."""
        return RayJobConfig(ray_address="ray://localhost:10001")

    @pytest.fixture
    def minio_config(self):
        """Create a MinIOConfig for testing."""
        return MinIOConfig(
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            bucket="test-bucket",
        )

    @pytest.fixture
    def vector_config(self):
        """Create a VectorDBConfig for testing."""
        return VectorDBConfig(
            provider_type="qdrant",
            qdrant_url="http://qdrant:6333",
            collection="test-collection",
        )

    @pytest.fixture
    def embed_config(self):
        """Create an EmbeddingConfig for testing."""
        return EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

    def test_creates_pipeline_with_configs(
        self,
        mock_cluster_strategy,
        ray_config,
        minio_config,
        vector_config,
        embed_config,
    ):
        """Verify IngestionPipeline is created with all required configs.

        **Why this test is important:**

        - Pipeline requires multiple configuration objects to function
        - Misconfiguration leads to runtime failures in distributed environment
        - Ensures dependency injection pattern works correctly

        **What it tests:**

        - Pipeline stores cluster_strategy reference correctly
        - Pipeline stores ray_config reference correctly
        - Pipeline stores minio_config reference correctly
        """
        pipeline = IngestionPipeline(
            cluster_strategy=mock_cluster_strategy,
            ray_config=ray_config,
            minio_config=minio_config,
            vector_config=vector_config,
            embed_config=embed_config,
        )
        assert pipeline.cluster_strategy is mock_cluster_strategy
        assert pipeline.ray_config is ray_config
        assert pipeline.minio_config is minio_config

    def test_from_env_creates_pipeline(self, mock_cluster_strategy):
        """Verify from_env() creates pipeline from environment variables.

        **Why this test is important:**

        - from_env() is the primary factory method for production use
        - Environment-based configuration is critical for containerized deployments
        - Namespace parameter enables multi-tenant configurations

        **What it tests:**

        - from_env() calls all config from_env() methods with namespace
        - Resulting pipeline has correct cluster_strategy reference
        - MinIOConfig.from_env() receives the namespace parameter
        - VectorDBConfig.from_env() receives the namespace parameter
        """
        with patch("core.ingestion.pipeline.MinIOConfig.from_env") as mock_minio:
            with patch("core.ingestion.pipeline.VectorDBConfig.from_env") as mock_vector:
                with patch("core.ingestion.pipeline.EmbeddingConfig.from_env") as mock_embed:
                    with patch("core.ingestion.pipeline.RayJobConfig.from_env") as mock_ray:
                        mock_minio.return_value = MagicMock(spec=MinIOConfig)
                        mock_vector.return_value = MagicMock(spec=VectorDBConfig)
                        mock_embed.return_value = MagicMock(spec=EmbeddingConfig)
                        mock_ray.return_value = MagicMock(spec=RayJobConfig)

                        pipeline = IngestionPipeline.from_env(
                            cluster_strategy=mock_cluster_strategy,
                            namespace="test-ns",
                        )

                        assert pipeline.cluster_strategy is mock_cluster_strategy
                        mock_minio.assert_called_once_with("test-ns")
                        mock_vector.assert_called_once_with("test-ns")


class TestIngestionPipelineRun:
    """Tests for IngestionPipeline.run() method."""

    @pytest.fixture
    def mock_cluster_strategy(self):
        """Create a mock cluster strategy."""
        strategy = MagicMock()
        strategy.init = MagicMock()
        strategy.shutdown = MagicMock()
        return strategy

    @pytest.fixture
    def mock_pipeline(self, mock_cluster_strategy):
        """Create a mock pipeline for testing run()."""
        ray_config = RayJobConfig(
            ray_address="ray://localhost:10001",
            s3_batch_size=10,
            checkpoint_enabled=False,
        )
        minio_config = MinIOConfig(
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            bucket="test-bucket",
        )
        vector_config = VectorDBConfig(
            provider_type="qdrant",
            qdrant_url="http://qdrant:6333",
            collection="test-collection",
        )
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        return IngestionPipeline(
            cluster_strategy=mock_cluster_strategy,
            ray_config=ray_config,
            minio_config=minio_config,
            vector_config=vector_config,
            embed_config=embed_config,
        )

    def test_run_initializes_and_shuts_down_cluster(self, mock_pipeline, mock_ray):
        """Verify run() initializes cluster at start and shuts down at end.

        **Why this test is important:**

        - Cluster resources are expensive and must be properly managed
        - Failure to initialize prevents job execution
        - Failure to shutdown causes resource leaks in Ray cluster

        **What it tests:**

        - cluster_strategy.init() is called exactly once at start
        - cluster_strategy.shutdown() is called exactly once at end
        - Method returns JobResult even for empty input
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        with patch("core.ingestion.pipeline.S3ClientWrapper", return_value=mock_s3):
            result = mock_pipeline.run(s3_prefix="inputs/", content_type="text")

        mock_pipeline.cluster_strategy.init.assert_called_once()
        mock_pipeline.cluster_strategy.shutdown.assert_called_once()
        assert result.total == 0

    def test_run_returns_job_result_on_empty_keys(self, mock_pipeline, mock_ray):
        """Verify run() returns JobResult with zeros when no keys found.

        **Why this test is important:**

        - Empty S3 prefixes are a valid use case (new bucket, cleanup)
        - Must return valid JobResult, not None or exception
        - Enables callers to distinguish between failure and empty input

        **What it tests:**

        - Return type is JobResult instance
        - total, successful, and failed are all zero
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        with patch("core.ingestion.pipeline.S3ClientWrapper", return_value=mock_s3):
            result = mock_pipeline.run(s3_prefix="inputs/", content_type="text")

        assert isinstance(result, JobResult)
        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0

    def test_run_shuts_down_cluster_on_exception(self, mock_pipeline, mock_ray):
        """Verify run() shuts down cluster even when exception occurs.

        **Why this test is important:**

        - Resource cleanup must happen regardless of execution outcome
        - Prevents orphaned Ray actors and memory leaks
        - Critical for production reliability and cost management

        **What it tests:**

        - Exception from S3 client propagates to caller
        - cluster_strategy.shutdown() is still called despite exception
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects.side_effect = Exception("S3 error")

        with patch("core.ingestion.pipeline.S3ClientWrapper", return_value=mock_s3):
            with pytest.raises(Exception, match="S3 error"):
                mock_pipeline.run(s3_prefix="inputs/", content_type="text")

        mock_pipeline.cluster_strategy.shutdown.assert_called_once()

    def test_run_processes_keys_with_batching(self, mock_pipeline, mock_ray):
        """Verify run() batches keys and submits ray tasks.

        **Why this test is important:**

        - Batching is essential for efficient distributed processing
        - Ray task submission must correctly aggregate results
        - Success/failure counting must be accurate for reporting

        **What it tests:**

        - Multiple keys are processed in batches
        - Results are correctly counted (2 successful, 1 failed)
        - Total matches the number of input keys
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = ["file1.txt", "file2.txt", "file3.txt"]

        # Mock ray functions
        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [
            [("file1.txt", True, ""), ("file2.txt", True, ""), ("file3.txt", False, "error")]
        ]

        with patch("core.ingestion.pipeline.S3ClientWrapper", return_value=mock_s3):
            with patch("core.ingestion.pipeline.RateLimiterActor") as mock_rate:
                mock_rate.remote.return_value = MagicMock()
                with patch("core.ingestion.pipeline.process_s3_batch_ray") as mock_task:
                    mock_task.options.return_value.remote.return_value = MagicMock()
                    result = mock_pipeline.run(s3_prefix="inputs/", content_type="text")

        assert result.successful == 2
        assert result.failed == 1
        assert result.total == 3


class TestIngestionPipelineExecute:
    """Tests for _execute method internals."""

    @pytest.fixture
    def mock_cluster_strategy(self):
        """Create a mock cluster strategy."""
        strategy = MagicMock()
        strategy.init = MagicMock()
        strategy.shutdown = MagicMock()
        return strategy

    @pytest.fixture
    def pipeline(self, mock_cluster_strategy):
        """Create pipeline for testing."""
        ray_config = RayJobConfig(
            ray_address="ray://localhost:10001",
            s3_batch_size=10,
            checkpoint_enabled=True,
            checkpoint_dir="/tmp/checkpoints",
        )
        minio_config = MinIOConfig(
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            bucket="test-bucket",
        )
        vector_config = VectorDBConfig(
            provider_type="qdrant",
            qdrant_url="http://qdrant:6333",
            collection="test",
        )
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        return IngestionPipeline(
            cluster_strategy=mock_cluster_strategy,
            ray_config=ray_config,
            minio_config=minio_config,
            vector_config=vector_config,
            embed_config=embed_config,
        )

    def test_create_batches_splits_correctly(self, pipeline):
        """Verify _create_batches splits keys into correct batch sizes.

        **Why this test is important:**

        - Batch size affects Ray task granularity and performance
        - Incorrect batching leads to OOM errors or underutilization
        - Last batch must handle remainder correctly

        **What it tests:**

        - 5 keys with batch_size=2 produces 3 batches
        - First two batches have exactly 2 items each
        - Last batch contains the remaining 1 item
        """
        keys = ["key1", "key2", "key3", "key4", "key5"]
        pipeline.ray_config = RayJobConfig(
            ray_address="ray://localhost:10001",
            s3_batch_size=2,
        )
        batches = pipeline._create_batches(keys)
        assert len(batches) == 3
        assert batches[0] == ["key1", "key2"]
        assert batches[1] == ["key3", "key4"]
        assert batches[2] == ["key5"]

    def test_create_batches_handles_empty_list(self, pipeline):
        """Verify _create_batches handles empty input.

        **Why this test is important:**

        - Empty input is a valid edge case that must not raise exceptions
        - Downstream code expects a list, not None or error
        - Prevents unnecessary Ray task submission for empty workloads

        **What it tests:**

        - Empty input list returns empty batch list
        - No exception is raised
        """
        batches = pipeline._create_batches([])
        assert batches == []

    def test_load_checkpoint_returns_empty_when_disabled(self, pipeline, mock_cluster_strategy):
        """Verify _load_checkpoint returns empty set when checkpointing disabled.

        **Why this test is important:**

        - Checkpointing is optional and must not interfere when disabled
        - Empty set ensures all keys are processed fresh
        - None checkpoint_path signals no checkpoint file to update

        **What it tests:**

        - Processed set is empty when checkpoint_enabled=False
        - checkpoint_path is None when checkpointing disabled
        """
        pipeline.ray_config = RayJobConfig(
            ray_address="ray://localhost:10001",
            checkpoint_enabled=False,
        )
        mock_s3 = MagicMock()
        processed, checkpoint_path = pipeline._load_checkpoint(mock_s3, ["key1"])
        assert processed == set()
        assert checkpoint_path is None


class TestIngestionPipelineCheckpoint:
    """Tests for checkpoint loading and saving."""

    @pytest.fixture
    def mock_cluster_strategy(self):
        """Create a mock cluster strategy."""
        strategy = MagicMock()
        return strategy

    @pytest.fixture
    def pipeline(self, mock_cluster_strategy):
        """Create pipeline with mocked configs."""
        ray_config = RayJobConfig(
            ray_address="ray://localhost:10001",
            checkpoint_enabled=True,
            checkpoint_dir="/tmp/checkpoints",
        )
        minio_config = MinIOConfig(
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            bucket="test-bucket",
        )
        vector_config = VectorDBConfig(
            provider_type="qdrant",
            qdrant_url="http://qdrant:6333",
            collection="test",
        )
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        return IngestionPipeline(
            cluster_strategy=mock_cluster_strategy,
            ray_config=ray_config,
            minio_config=minio_config,
            vector_config=vector_config,
            embed_config=embed_config,
        )

    def test_load_checkpoint_returns_empty_set_when_no_checkpoint(self, pipeline):
        """Verify _load_checkpoint returns empty set when checkpoint does not exist.

        **Why this test is important:**

        - First run has no existing checkpoint file
        - Must return empty set to process all keys
        - checkpoint_path must be set for saving progress later

        **What it tests:**

        - Empty set returned when CheckpointManager.load() returns empty
        - checkpoint_path is not None (ready for saving)
        """
        with patch("core.ingestion.pipeline.CheckpointManager") as mock_checkpoint_class:
            mock_manager = MagicMock()
            mock_manager.load.return_value = set()
            mock_checkpoint_class.return_value = mock_manager

            mock_s3 = MagicMock()
            processed, checkpoint_path = pipeline._load_checkpoint(mock_s3, ["key1"])

        assert processed == set()
        assert checkpoint_path is not None

    def test_load_checkpoint_returns_set_from_file(self, pipeline):
        """Verify _load_checkpoint returns set of previously processed keys.

        **Why this test is important:**

        - Resume capability depends on correct checkpoint loading
        - Skipping already-processed keys saves time and resources
        - Ensures idempotent re-runs after failures

        **What it tests:**

        - Set of processed keys matches what CheckpointManager returns
        - All three checkpoint keys are present in returned set
        """
        with patch("core.ingestion.pipeline.CheckpointManager") as mock_checkpoint_class:
            mock_manager = MagicMock()
            mock_manager.load.return_value = {"key1", "key2", "key3"}
            mock_checkpoint_class.return_value = mock_manager

            mock_s3 = MagicMock()
            processed, checkpoint_path = pipeline._load_checkpoint(mock_s3, ["key1"])

        assert processed == {"key1", "key2", "key3"}


class TestIngestionPipelineContentType:
    """Tests for content type handling."""

    @pytest.fixture
    def mock_cluster_strategy(self):
        """Create a mock cluster strategy."""
        strategy = MagicMock()
        strategy.init = MagicMock()
        strategy.shutdown = MagicMock()
        return strategy

    @pytest.fixture
    def pipeline(self, mock_cluster_strategy):
        """Create pipeline for testing."""
        ray_config = RayJobConfig(
            ray_address="ray://localhost:10001",
            s3_batch_size=10,
            checkpoint_enabled=False,
        )
        minio_config = MinIOConfig(
            endpoint_url="http://minio:9000",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            bucket="test-bucket",
        )
        vector_config = VectorDBConfig(
            provider_type="qdrant",
            qdrant_url="http://qdrant:6333",
            collection="test-collection",
        )
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        return IngestionPipeline(
            cluster_strategy=mock_cluster_strategy,
            ray_config=ray_config,
            minio_config=minio_config,
            vector_config=vector_config,
            embed_config=embed_config,
        )

    def test_run_uses_text_task_for_text_content(self, pipeline, mock_ray):
        """Verify run() uses process_s3_batch_ray for text content.

        **Why this test is important:**

        - Text and image content require different processing pipelines
        - Using wrong task type leads to processing failures
        - Ensures content_type parameter controls task selection

        **What it tests:**

        - process_s3_batch_ray is called for content_type="text"
        - process_s3_image_batch_ray is NOT called for text content
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = ["file.txt"]
        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("file.txt", True, "")]]

        with patch("core.ingestion.pipeline.S3ClientWrapper", return_value=mock_s3):
            with patch("core.ingestion.pipeline.RateLimiterActor") as mock_rate:
                mock_rate.remote.return_value = MagicMock()
                with patch("core.ingestion.pipeline.process_s3_batch_ray") as mock_text_task:
                    with patch("core.ingestion.pipeline.process_s3_image_batch_ray") as mock_image_task:
                        mock_text_task.options.return_value.remote.return_value = MagicMock()
                        pipeline.run(s3_prefix="inputs/", content_type="text")

                        mock_text_task.options.assert_called_once()
                        mock_image_task.options.assert_not_called()

    def test_run_uses_image_task_for_image_content(self, pipeline, mock_ray):
        """Verify run() uses process_s3_image_batch_ray for image content.

        **Why this test is important:**

        - Image processing requires specialized embedding and chunking
        - Mixing up task types causes data corruption or failures
        - Validates the content_type routing logic works bidirectionally

        **What it tests:**

        - process_s3_image_batch_ray is called for content_type="images"
        - process_s3_batch_ray is NOT called for image content
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = ["file.jpg"]
        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("file.jpg", True, "")]]

        with patch("core.ingestion.pipeline.S3ClientWrapper", return_value=mock_s3):
            with patch("core.ingestion.pipeline.RateLimiterActor") as mock_rate:
                mock_rate.remote.return_value = MagicMock()
                with patch("core.ingestion.pipeline.process_s3_batch_ray") as mock_text_task:
                    with patch("core.ingestion.pipeline.process_s3_image_batch_ray") as mock_image_task:
                        mock_image_task.options.return_value.remote.return_value = MagicMock()
                        pipeline.run(s3_prefix="images/", content_type="images")

                        mock_image_task.options.assert_called_once()
                        mock_text_task.options.assert_not_called()

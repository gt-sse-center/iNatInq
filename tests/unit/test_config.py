"""Unit tests for config module.

This file tests the configuration classes and their from_env() methods,
particularly the RayJobConfig.dashboard_address behavior for different
environments.

# Test Coverage

The tests cover:
  - RayJobConfig: dashboard_address auto-detection for K8s, Docker Compose, and local
  - MinIOConfig: S3/MinIO configuration including resilience settings
  - Environment variable parsing
  - Default value handling

# Running Tests

Run with: pytest tests/unit/test_config.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from pydantic import ValidationError

from config import (
    DatabricksRayJobConfig,
    EmbeddingConfig,
    INatConfig,
    MinIOConfig,
    ProviderType,
    RayJobConfig,
    SemanticCacheConfig,
    Settings,
    VectorDBConfig,
    resolve_vector_db_provider,
)


# =============================================================================
# RayJobConfig Tests
# =============================================================================


class TestRayJobConfigDashboardAddress:
    """Test suite for RayJobConfig.dashboard_address auto-detection."""

    @patch.dict(
        os.environ,
        {"RAY_DASHBOARD_ADDRESS": "http://custom-ray:8265", "K8S_NAMESPACE": "ml-system"},
        clear=False,
    )
    def test_explicit_dashboard_address_takes_precedence(self) -> None:
        """Test that explicit RAY_DASHBOARD_ADDRESS takes precedence.

        **Why this test is important:**
          - Explicit configuration should always win
          - Allows overriding auto-detection
          - Critical for custom deployments

        **What it tests:**
          - RAY_DASHBOARD_ADDRESS env var is used
          - K8S_NAMESPACE is ignored when explicit address is set
        """
        config = RayJobConfig.from_env()
        assert config.dashboard_address == "http://custom-ray:8265"

    @patch.dict(
        os.environ,
        {"K8S_NAMESPACE": "prod-ml", "PIPELINE_ENV": "cluster"},
        clear=False,
    )
    @patch("config.os.environ.get")
    def test_kubernetes_namespace_based_address(self, mock_get: patch) -> None:
        """Test that K8S_NAMESPACE is used for dashboard address in Kubernetes.

        **Why this test is important:**
          - Kubernetes uses namespace-based DNS
          - Service discovery works via ray-head.{namespace}:8265
          - Critical for in-cluster deployments

        **What it tests:**
          - Dashboard address is constructed from K8S_NAMESPACE
          - Format is http://ray-head.{namespace}:8265
        """

        def environ_get(key: str, default: str | None = None) -> str | None:
            env_map = {
                "RAY_DASHBOARD_ADDRESS": None,
                "K8S_NAMESPACE": "prod-ml",
                "RAY_ADDRESS": None,
                "PIPELINE_ENV": "cluster",
            }
            return env_map.get(key, default)

        mock_get.side_effect = environ_get

        # Need to also patch _is_in_cluster to return False (no service account)
        with patch("config._is_in_cluster", return_value=False):
            config = RayJobConfig.from_env()
            assert config.dashboard_address == "http://ray-head.prod-ml:8265"

    @patch.dict(
        os.environ,
        {"PIPELINE_ENV": "local"},
        clear=False,
    )
    @patch("config.os.environ.get")
    def test_docker_compose_style_address(self, mock_get: patch) -> None:
        """Test that PIPELINE_ENV=local uses simple hostname for Docker Compose.

        **Why this test is important:**
          - Docker Compose uses simple service names
          - ray-head:8265 works without namespace
          - Critical for local development

        **What it tests:**
          - Dashboard address is http://ray-head:8265
          - No namespace is included
        """

        def environ_get(key: str, default: str | None = None) -> str | None:
            env_map = {
                "RAY_DASHBOARD_ADDRESS": None,
                "K8S_NAMESPACE": None,
                "RAY_ADDRESS": None,
                "PIPELINE_ENV": "local",
            }
            return env_map.get(key, default)

        mock_get.side_effect = environ_get

        with patch("config._is_in_cluster", return_value=False):
            config = RayJobConfig.from_env()
            assert config.dashboard_address == "http://ray-head:8265"

    @patch.dict(
        os.environ,
        {},
        clear=False,
    )
    @patch("config.os.environ.get")
    def test_local_development_without_pipeline_env(self, mock_get: patch) -> None:
        """Test that local development falls back to localhost.

        **Why this test is important:**
          - Without PIPELINE_ENV, should fall back to localhost
          - Works for running Ray locally outside Docker
          - Critical for development flexibility

        **What it tests:**
          - Dashboard address is http://localhost:8265
          - Works when no environment indicators are set
        """

        def environ_get(key: str, default: str | None = None) -> str | None:
            env_map = {
                "RAY_DASHBOARD_ADDRESS": None,
                "K8S_NAMESPACE": None,
                "RAY_ADDRESS": None,
                "PIPELINE_ENV": None,
            }
            return env_map.get(key, default)

        mock_get.side_effect = environ_get

        with patch("config._is_in_cluster", return_value=False):
            config = RayJobConfig.from_env()
            assert config.dashboard_address == "http://localhost:8265"

    @patch.dict(
        os.environ,
        {"RAY_DASHBOARD_ADDRESS": "http://ray-head:8265", "RAY_ADDRESS": "ray://ray-head:20001"},
        clear=False,
    )
    def test_both_addresses_can_be_set(self) -> None:
        """Test that both ray_address and dashboard_address can be set.

        **Why this test is important:**
          - Ray client protocol (ray://) and HTTP dashboard are different
          - Both may be needed for different operations
          - Critical for complete functionality

        **What it tests:**
          - ray_address is set to Ray client protocol address
          - dashboard_address is set to HTTP dashboard address
          - Both are independent
        """
        config = RayJobConfig.from_env()
        assert config.ray_address == "ray://ray-head:20001"
        assert config.dashboard_address == "http://ray-head:8265"


class TestRayJobConfigOtherSettings:
    """Test suite for other RayJobConfig settings."""

    @patch.dict(
        os.environ,
        {
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "RAY_NUM_WORKERS": "4",
            "RAY_WORKER_CPUS": "2.0",
            "RAY_NAMESPACE": "custom-ns",
        },
        clear=False,
    )
    def test_other_settings_are_parsed(self) -> None:
        """Test that other RayJobConfig settings are parsed correctly.

        **Why this test is important:**
          - All settings should be parsed from environment
          - Type conversion should work correctly
          - Critical for complete functionality

        **What it tests:**
          - num_workers is parsed as int
          - worker_cpus is parsed as float
          - ray_namespace is parsed as string
        """
        config = RayJobConfig.from_env()
        assert config.num_workers == 4
        assert config.worker_cpus == 2.0
        assert config.ray_namespace == "custom-ns"

    @patch.dict(
        os.environ,
        {"RAY_DASHBOARD_ADDRESS": "http://ray-head:8265"},
        clear=False,
    )
    def test_default_values(self) -> None:
        """Test that default values are used when env vars are not set.

        **Why this test is important:**
          - Sensible defaults reduce configuration burden
          - Critical for easy onboarding

        **What it tests:**
          - Default values are applied
          - num_workers defaults to 0 (auto-scale)
          - checkpoint_enabled defaults to True
        """
        config = RayJobConfig.from_env()
        assert config.embed_batch_min == 1
        assert config.embed_batch_max == 8
        assert config.batch_upsert_size == 200
        assert config.checkpoint_enabled is True


# =============================================================================
# DatabricksRayJobConfig Tests
# =============================================================================


class TestDatabricksRayJobConfig:
    """Test suite for DatabricksRayJobConfig parsing and validation."""

    @pytest.mark.parametrize(
        "missing_key",
        ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_JOB_ID"],
    )
    def test_missing_required_fields(self, missing_key: str) -> None:
        """Test that required env vars are validated."""
        base_env = {
            "DATABRICKS_HOST": "https://dbc.example.cloud",
            "DATABRICKS_TOKEN": "databricks-token",
            "DATABRICKS_JOB_ID": "123",
        }
        base_env.pop(missing_key)

        with patch.dict(os.environ, base_env, clear=True):
            with pytest.raises(ValueError, match="Missing required Databricks config"):
                DatabricksRayJobConfig.from_env()

    def test_default_task_type(self) -> None:
        """Test that task_type defaults to python."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "456",
            },
            clear=True,
        ):
            config = DatabricksRayJobConfig.from_env()
            assert config.task_type == "python"
            assert config.job_id == 456

    def test_invalid_task_type(self) -> None:
        """Test that invalid task types are rejected."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_TASK_TYPE": "jar",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Invalid DATABRICKS_TASK_TYPE"):
                DatabricksRayJobConfig.from_env()

    def test_parses_optional_inat_job_id(self) -> None:
        """Test optional DATABRICKS_INAT_JOB_ID parsing."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_INAT_JOB_ID": "12345",
            },
            clear=True,
        ):
            config = DatabricksRayJobConfig.from_env()
            assert config.job_id == 789
            assert config.inat_job_id == 12345

    def test_invalid_inat_job_id(self) -> None:
        """Test invalid optional iNat job id is rejected."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_INAT_JOB_ID": "not-an-int",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="DATABRICKS_INAT_JOB_ID must be an integer"):
                DatabricksRayJobConfig.from_env()

    def test_parses_optional_s3_autoloader_job_id(self) -> None:
        """Test optional DATABRICKS_S3_AUTOLOADER_JOB_ID parsing."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_S3_AUTOLOADER_JOB_ID": "45678",
            },
            clear=True,
        ):
            config = DatabricksRayJobConfig.from_env()
            assert config.s3_autoloader_job_id == 45678

    def test_invalid_s3_autoloader_job_id(self) -> None:
        """Test invalid optional s3 autoloader job id is rejected."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_S3_AUTOLOADER_JOB_ID": "not-an-int",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="DATABRICKS_S3_AUTOLOADER_JOB_ID must be an integer"):
                DatabricksRayJobConfig.from_env()

    def test_parses_optional_s3_bronze_job_id(self) -> None:
        """Test optional DATABRICKS_FROM_BRONZE_JOB_ID parsing."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_FROM_BRONZE_JOB_ID": "99999",
            },
            clear=True,
        ):
            config = DatabricksRayJobConfig.from_env()
            assert config.s3_bronze_job_id == 99999

    def test_invalid_s3_bronze_job_id(self) -> None:
        """Test invalid optional s3 bronze job id is rejected."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_JOB_ID": "789",
                "DATABRICKS_FROM_BRONZE_JOB_ID": "not-an-int",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="DATABRICKS_FROM_BRONZE_JOB_ID must be an integer"):
                DatabricksRayJobConfig.from_env()

    def test_from_env_allows_missing_job_id_when_not_required(self) -> None:
        """CDC producer path should not require DATABRICKS_JOB_ID."""
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://dbc.example.cloud",
                "DATABRICKS_TOKEN": "databricks-token",
                "DATABRICKS_S3_AUTOLOADER_JOB_ID": "45678",
            },
            clear=True,
        ):
            config = DatabricksRayJobConfig.from_env(require_job_id=False)
            assert config.job_id is None
            assert config.s3_autoloader_job_id == 45678


# =============================================================================
# MinIOConfig Tests
# =============================================================================


class TestMinIOConfigResilience:
    """Test suite for MinIOConfig resilience settings."""

    @patch.dict(
        os.environ,
        {
            "S3_ENDPOINT": "http://minio:9000",
            "S3_TIMEOUT": "60",
            "S3_MAX_RETRIES": "5",
            "S3_RETRY_MIN_WAIT": "2.0",
            "S3_RETRY_MAX_WAIT": "20.0",
            "S3_CIRCUIT_BREAKER_THRESHOLD": "10",
            "S3_CIRCUIT_BREAKER_TIMEOUT": "300",
        },
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_resilience_settings_from_env(self, mock_cluster: patch) -> None:
        """Test that resilience settings are parsed from environment.

        **Why this test is important:**
          - Resilience settings must be configurable per environment
          - Production may need different timeouts than development
          - Critical for operational flexibility

        **What it tests:**
          - timeout is parsed from S3_TIMEOUT
          - max_retries is parsed from S3_MAX_RETRIES
          - retry_min_wait is parsed from S3_RETRY_MIN_WAIT
          - retry_max_wait is parsed from S3_RETRY_MAX_WAIT
          - circuit_breaker_threshold is parsed from S3_CIRCUIT_BREAKER_THRESHOLD
          - circuit_breaker_timeout is parsed from S3_CIRCUIT_BREAKER_TIMEOUT
        """
        config = MinIOConfig.from_env()

        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.retry_min_wait == 2.0
        assert config.retry_max_wait == 20.0
        assert config.circuit_breaker_threshold == 10
        assert config.circuit_breaker_timeout == 300

    @patch.dict(
        os.environ,
        {"S3_ENDPOINT": "http://minio:9000"},
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_resilience_default_values(self, mock_cluster: patch) -> None:
        """Test that default resilience values are applied.

        **Why this test is important:**
          - Sensible defaults reduce configuration burden
          - Critical for easy onboarding
          - Defaults should match common production patterns

        **What it tests:**
          - Default timeout is 30 seconds
          - Default max_retries is 3
          - Default retry_min_wait is 1.0 seconds
          - Default retry_max_wait is 10.0 seconds
          - Default circuit_breaker_threshold is 5
          - Default circuit_breaker_timeout is 120 seconds
        """
        config = MinIOConfig.from_env()

        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_min_wait == 1.0
        assert config.retry_max_wait == 10.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 120


# =============================================================================
# EmbeddingConfig Tests
# =============================================================================


class TestEmbeddingConfigGuardrails:
    """Test suite for hosted/local CLIP provider guardrails."""

    @patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "clip",
            "CLIP_URL": "https://example.eastus.inference.ml.azure.com/score",
            "CLIP_API_KEY": "test-key",
            "CLIP_MODEL": "clip-vit-base-patch32",
        },
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_local_clip_with_score_url_and_key_raises(self, mock_cluster: patch) -> None:
        """Fail fast on likely hosted endpoint misconfiguration."""
        with pytest.raises(ValueError, match="Likely CLIP provider mismatch"):
            EmbeddingConfig.from_env()

    @patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "hosted_clip",
            "CLIP_URL": "https://example.eastus.inference.ml.azure.com/score",
            "CLIP_API_KEY": "test-key",
            "CLIP_MODEL": "clip-vit-base-patch32",
        },
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_hosted_clip_with_score_url_is_allowed(self, mock_cluster: patch) -> None:
        """Hosted mode should accept /score endpoints."""
        config = EmbeddingConfig.from_env()

        assert config.provider_type is ProviderType.HOSTED_CLIP
        assert config.clip_url == "https://example.eastus.inference.ml.azure.com/score"


# =============================================================================
# VectorDBConfig Tests
# =============================================================================


class TestResolveVectorDBProvider:
    """Test suite for vector DB provider resolution."""

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_qdrant_when_unset(self) -> None:
        """Resolver returns qdrant when provider is not configured."""
        assert resolve_vector_db_provider() == "qdrant"

    @patch.dict(os.environ, {"VECTOR_DB_PROVIDER": "  QdRaNt  "}, clear=True)
    def test_normalizes_provider_from_env(self) -> None:
        """Resolver normalizes casing and whitespace from env values."""
        assert resolve_vector_db_provider() == "qdrant"

    def test_uses_explicit_provider_override(self) -> None:
        """Resolver respects explicit provider overrides."""
        assert resolve_vector_db_provider(provider=" Qdrant ") == "qdrant"


class TestVectorDBConfigQdrantResilience:
    """Test suite for VectorDBConfig Qdrant resilience settings."""

    @patch.dict(
        os.environ,
        {
            "VECTOR_DB_PROVIDER": "qdrant",
            "QDRANT_URL": "http://qdrant:6333",
            "QDRANT_TIMEOUT": "600",
            "QDRANT_CIRCUIT_BREAKER_THRESHOLD": "10",
            "QDRANT_CIRCUIT_BREAKER_TIMEOUT": "120",
        },
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_qdrant_resilience_settings_from_env(self, mock_cluster: patch) -> None:
        """Test that Qdrant resilience settings are parsed from environment.

        **Why this test is important:**
          - Resilience settings must be configurable per environment
          - Production may need different timeouts than development
          - Critical for operational flexibility

        **What it tests:**
          - qdrant_timeout is parsed from QDRANT_TIMEOUT
          - qdrant_circuit_breaker_threshold is parsed correctly
          - qdrant_circuit_breaker_timeout is parsed correctly
        """
        from config import VectorDBConfig

        config = VectorDBConfig.from_env()

        assert config.qdrant_timeout == 600
        assert config.qdrant_circuit_breaker_threshold == 10
        assert config.qdrant_circuit_breaker_timeout == 120

    @patch.dict(
        os.environ,
        {
            "VECTOR_DB_PROVIDER": "qdrant",
            "QDRANT_URL": "http://qdrant:6333",
        },
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_qdrant_resilience_default_values(self, mock_cluster: patch) -> None:
        """Test that default Qdrant resilience values are applied.

        **Why this test is important:**
          - Sensible defaults reduce configuration burden
          - Critical for easy onboarding
          - Defaults should match common production patterns

        **What it tests:**
          - Default qdrant_timeout is 300 seconds
          - Default qdrant_circuit_breaker_threshold is 3
          - Default qdrant_circuit_breaker_timeout is 60 seconds
        """
        from config import VectorDBConfig

        config = VectorDBConfig.from_env()

        assert config.qdrant_timeout == 300
        assert config.qdrant_circuit_breaker_threshold == 3
        assert config.qdrant_circuit_breaker_timeout == 60

    @patch.dict(
        os.environ,
        {
            "VECTOR_DB_PROVIDER": "   ",
        },
        clear=False,
    )
    @patch("config._is_in_cluster", return_value=False)
    def test_blank_provider_defaults_to_qdrant(self, mock_cluster: patch) -> None:
        """Test that blank provider values fall back to qdrant default."""
        from config import VectorDBConfig

        config = VectorDBConfig.from_env()
        assert config.provider_type == "qdrant"


# =============================================================================
# RayJobConfig Image Pipeline Fields Tests
# =============================================================================


class TestRayJobConfigImagePipelineFields:
    """Test suite for RayJobConfig image pipeline fields (s3_prefix, image_max_items, image_page_size)."""

    @patch.dict(
        os.environ,
        {
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "S3_PREFIX": "images/birds/",
            "IMAGE_MAX_ITEMS": "500",
            "IMAGE_PAGE_SIZE": "2000",
        },
        clear=False,
    )
    def test_image_pipeline_fields_from_env(self) -> None:
        """Test that image pipeline fields are parsed from environment."""
        config = RayJobConfig.from_env()
        assert config.s3_prefix == "images/birds/"
        assert config.image_max_items == 500
        assert config.image_page_size == 2000

    @patch.dict(
        os.environ,
        {"RAY_DASHBOARD_ADDRESS": "http://ray-head:8265"},
        clear=False,
    )
    def test_image_pipeline_default_values(self) -> None:
        """Test that image pipeline fields use correct defaults."""
        config = RayJobConfig.from_env()
        assert config.s3_prefix == ""
        assert config.image_max_items is None
        assert config.image_page_size == 1000

    @patch.dict(
        os.environ,
        {
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "IMAGE_MAX_ITEMS": "0",
        },
        clear=False,
    )
    def test_image_max_items_zero_returns_none(self) -> None:
        """Test that IMAGE_MAX_ITEMS=0 is treated as None (no limit)."""
        config = RayJobConfig.from_env()
        assert config.image_max_items is None

    @patch.dict(
        os.environ,
        {
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "IMAGE_MAX_ITEMS": "-5",
        },
        clear=False,
    )
    def test_image_max_items_negative_returns_none(self) -> None:
        """Test that negative IMAGE_MAX_ITEMS is treated as None."""
        config = RayJobConfig.from_env()
        assert config.image_max_items is None

    @patch.dict(
        os.environ,
        {
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "IMAGE_MAX_ITEMS": "not-a-number",
        },
        clear=False,
    )
    def test_image_max_items_invalid_returns_none(self) -> None:
        """Test that non-integer IMAGE_MAX_ITEMS is treated as None."""
        config = RayJobConfig.from_env()
        assert config.image_max_items is None


# =============================================================================
# INatConfig Tests
# =============================================================================


class TestINatConfig:
    """Test suite for INatConfig from_env() parsing and validation."""

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_max_rows_raises(self) -> None:
        """Test that missing INAT_MAX_ROWS raises RuntimeError."""
        with pytest.raises(RuntimeError, match="INAT_MAX_ROWS is required"):
            INatConfig.from_env()

    @patch.dict(os.environ, {"INAT_MAX_ROWS": "0"}, clear=True)
    def test_zero_max_rows_raises(self) -> None:
        """Test that INAT_MAX_ROWS=0 raises RuntimeError."""
        with pytest.raises(RuntimeError, match="INAT_MAX_ROWS must be a positive integer"):
            INatConfig.from_env()

    @patch.dict(os.environ, {"INAT_MAX_ROWS": "-10"}, clear=True)
    def test_negative_max_rows_raises(self) -> None:
        """Test that negative INAT_MAX_ROWS raises RuntimeError."""
        with pytest.raises(RuntimeError, match="INAT_MAX_ROWS must be a positive integer"):
            INatConfig.from_env()

    @patch.dict(os.environ, {"INAT_MAX_ROWS": "100"}, clear=True)
    def test_default_values(self) -> None:
        """Test that INatConfig defaults are applied correctly."""
        config = INatConfig.from_env()
        assert config.image_size == "medium"
        assert config.max_rows == 100
        assert config.metadata_url == ""
        assert config.photo_base_url == "https://inaturalist-open-data.s3.amazonaws.com/photos"
        assert config.timeout_s == 120
        assert config.cb_failure_threshold == 5
        assert config.cb_recovery_timeout_s == 30
        assert config.image_max_items is None

    @patch.dict(
        os.environ,
        {
            "INAT_MAX_ROWS": "50",
            "INAT_IMAGE_SIZE": " Large ",
            "INAT_METADATA_URL": "https://example.com/photos.tsv",
            "INAT_PHOTO_BASE_URL": "https://custom.cdn.com/photos",
            "INAT_TIMEOUT_S": "300",
            "INAT_CB_FAILURE_THRESHOLD": "10",
            "INAT_CB_RECOVERY_TIMEOUT_S": "60",
            "IMAGE_MAX_ITEMS": "200",
        },
        clear=True,
    )
    def test_all_fields_from_env(self) -> None:
        """Test that all INatConfig fields are parsed from environment."""
        config = INatConfig.from_env()
        assert config.image_size == "large"
        assert config.max_rows == 50
        assert config.metadata_url == "https://example.com/photos.tsv"
        assert config.photo_base_url == "https://custom.cdn.com/photos"
        assert config.timeout_s == 300
        assert config.cb_failure_threshold == 10
        assert config.cb_recovery_timeout_s == 60
        assert config.image_max_items == 200

    @patch.dict(os.environ, {"INAT_MAX_ROWS": "  25  "}, clear=True)
    def test_whitespace_handling(self) -> None:
        """Test that INAT_MAX_ROWS whitespace is stripped."""
        config = INatConfig.from_env()
        assert config.max_rows == 25


# =============================================================================
# SemanticCacheConfig Tests
# =============================================================================


class TestSemanticCacheConfig:
    """Test suite for SemanticCacheConfig defaults, from_env, and validation."""

    def test_default_values(self) -> None:
        """All defaults: enabled=True, threshold=0.95, max_entries=1000, interval=3600, timeout=5."""
        config = SemanticCacheConfig()
        assert config.enabled is True
        assert config.similarity_threshold == 0.95
        assert config.max_entries_per_collection == 1000
        assert config.invalidation_interval_seconds == 3600
        assert config.timeout_s == 5

    @patch.dict(
        os.environ,
        {
            "SEMANTIC_CACHE_ENABLED": "false",
            "SEMANTIC_CACHE_SIMILARITY_THRESHOLD": "0.8",
            "SEMANTIC_CACHE_MAX_ENTRIES": "500",
            "SEMANTIC_CACHE_INVALIDATION_INTERVAL": "1800",
            "SEMANTIC_CACHE_TIMEOUT": "10",
        },
        clear=True,
    )
    def test_from_env_overrides(self) -> None:
        """Each SEMANTIC_CACHE_* env var correctly overrides its field."""
        config = SemanticCacheConfig.from_env()
        assert config.enabled is False
        assert config.similarity_threshold == 0.8
        assert config.max_entries_per_collection == 500
        assert config.invalidation_interval_seconds == 1800
        assert config.timeout_s == 10

    def test_invalid_threshold_raises(self) -> None:
        """Threshold outside [0.0, 1.0] raises ValidationError."""
        with pytest.raises(ValidationError, match="similarity_threshold"):
            SemanticCacheConfig(similarity_threshold=1.5)
        with pytest.raises(ValidationError, match="similarity_threshold"):
            SemanticCacheConfig(similarity_threshold=-0.1)

    def test_invalid_max_entries_raises(self) -> None:
        """max_entries_per_collection <= 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="max_entries_per_collection"):
            SemanticCacheConfig(max_entries_per_collection=0)
        with pytest.raises(ValidationError, match="max_entries_per_collection"):
            SemanticCacheConfig(max_entries_per_collection=-1)

    def test_settings_includes_semantic_cache(self) -> None:
        """Settings has semantic_cache field of type SemanticCacheConfig."""
        settings = Settings.model_construct(
            embedding=None,
            vector_db=None,
            minio=None,
            k8s_namespace="test",
            semantic_cache=SemanticCacheConfig(),
        )
        assert isinstance(settings.semantic_cache, SemanticCacheConfig)
        assert settings.semantic_cache.enabled is True

    @patch.dict(os.environ, {"SEMANTIC_CACHE_ENABLED": "false"}, clear=True)
    def test_disabled_via_env(self) -> None:
        """SEMANTIC_CACHE_ENABLED=false produces enabled=False."""
        config = SemanticCacheConfig.from_env()
        assert config.enabled is False

    def test_config_is_frozen(self) -> None:
        """SemanticCacheConfig instances are immutable."""
        config = SemanticCacheConfig()
        with pytest.raises(ValidationError):
            config.enabled = False  # type: ignore[misc]

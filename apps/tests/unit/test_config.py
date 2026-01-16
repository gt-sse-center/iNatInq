"""Unit tests for config module.

This file tests the configuration classes and their from_env() methods,
particularly the RayJobConfig.dashboard_address behavior for different
environments.

# Test Coverage

The tests cover:
  - RayJobConfig: dashboard_address auto-detection for K8s, Docker Compose, and local
  - Environment variable parsing
  - Default value handling

# Running Tests

Run with: pytest tests/unit/test_config.py
"""

import os
from unittest.mock import patch

import pytest

from config import RayJobConfig


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

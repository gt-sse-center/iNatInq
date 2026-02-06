"""Unit tests for config_loader module.

Tests cover YAML loading, deep merging, env var substitution,
and the apply-as-defaults bridging logic.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config_loader import (
    _resolve_env_substitutions,
    apply_yaml_defaults,
    deep_merge,
    initialize_config,
    load_yaml_config,
)


# =============================================================================
# deep_merge Tests
# =============================================================================


class TestDeepMerge:
    """Tests for deep_merge()."""

    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"top": {"a": 1, "b": 2}, "other": "x"}
        overlay = {"top": {"b": 3, "c": 4}}
        result = deep_merge(base, overlay)
        assert result == {"top": {"a": 1, "b": 3, "c": 4}, "other": "x"}

    def test_overlay_wins_on_type_conflict(self):
        base = {"key": {"nested": True}}
        overlay = {"key": "flat_string"}
        result = deep_merge(base, overlay)
        assert result == {"key": "flat_string"}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        deep_merge(base, overlay)
        assert base == {"a": {"x": 1}}
        assert overlay == {"a": {"y": 2}}

    def test_empty_overlay(self):
        base = {"a": 1}
        assert deep_merge(base, {}) == {"a": 1}

    def test_empty_base(self):
        overlay = {"a": 1}
        assert deep_merge({}, overlay) == {"a": 1}

    def test_deeply_nested(self):
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        overlay = {"l1": {"l2": {"l3": {"b": 99}}}}
        result = deep_merge(base, overlay)
        assert result["l1"]["l2"]["l3"] == {"a": 1, "b": 99}


# =============================================================================
# load_yaml_config Tests
# =============================================================================


class TestLoadYamlConfig:
    """Tests for load_yaml_config()."""

    def test_base_only(self, tmp_path):
        """Loads config.yaml when no env is specified."""
        base = {"storage": {"bucket": "test-bucket"}, "embeddings": {"provider": "ollama"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))

        result = load_yaml_config(env=None, config_dir=tmp_path)
        assert result["storage"]["bucket"] == "test-bucket"
        assert result["embeddings"]["provider"] == "ollama"

    def test_with_env_overlay(self, tmp_path):
        """Merges environment overlay on top of base."""
        base = {"storage": {"bucket": "default", "region": "us-east-1"}}
        overlay = {"storage": {"bucket": "dev-data"}}

        (tmp_path / "config.yaml").write_text(yaml.dump(base))
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        (env_dir / "dev.yaml").write_text(yaml.dump(overlay))

        result = load_yaml_config(env="dev", config_dir=tmp_path)
        assert result["storage"]["bucket"] == "dev-data"
        assert result["storage"]["region"] == "us-east-1"

    def test_with_secrets(self, tmp_path):
        """Merges secrets.yaml on top of base + env."""
        base = {"storage": {"bucket": "default"}}
        secrets = {"storage": {"access_key": "secret123"}}

        (tmp_path / "config.yaml").write_text(yaml.dump(base))
        (tmp_path / "secrets.yaml").write_text(yaml.dump(secrets))

        result = load_yaml_config(env=None, config_dir=tmp_path)
        assert result["storage"]["bucket"] == "default"
        assert result["storage"]["access_key"] == "secret123"

    def test_missing_base_config(self, tmp_path):
        """Returns empty dict when config.yaml doesn't exist."""
        result = load_yaml_config(env=None, config_dir=tmp_path)
        assert result == {}

    def test_missing_env_overlay_is_ok(self, tmp_path):
        """Silently skips missing environment overlay."""
        base = {"key": "value"}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))

        result = load_yaml_config(env="nonexistent", config_dir=tmp_path)
        assert result == {"key": "value"}

    def test_env_substitution(self, tmp_path):
        """Resolves ${VAR} patterns from os.environ."""
        base = {"storage": {"endpoint": "${MY_ENDPOINT}"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))

        with patch.dict(os.environ, {"MY_ENDPOINT": "http://custom:9000"}):
            result = load_yaml_config(env=None, config_dir=tmp_path)
            assert result["storage"]["endpoint"] == "http://custom:9000"


# =============================================================================
# _resolve_env_substitutions Tests
# =============================================================================


class TestResolveEnvSubstitutions:
    """Tests for _resolve_env_substitutions()."""

    def test_simple_substitution(self):
        with patch.dict(os.environ, {"FOO": "bar"}):
            assert _resolve_env_substitutions("${FOO}") == "bar"

    def test_unresolved_left_as_is(self):
        result = _resolve_env_substitutions("${DEFINITELY_NOT_SET_12345}")
        assert result == "${DEFINITELY_NOT_SET_12345}"

    def test_multiple_substitutions(self):
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            result = _resolve_env_substitutions("http://${HOST}:${PORT}")
            assert result == "http://localhost:8080"

    def test_no_substitution_needed(self):
        assert _resolve_env_substitutions("plain string") == "plain string"


# =============================================================================
# apply_yaml_defaults Tests
# =============================================================================


class TestApplyYamlDefaults:
    """Tests for apply_yaml_defaults()."""

    def test_sets_unset_vars(self):
        """Sets env vars from YAML when not already set."""
        config = {"storage": {"bucket": "yaml-bucket", "region": "eu-west-1"}}

        env_to_clear = ["S3_BUCKET", "S3_REGION"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_to_clear}

        with patch.dict(os.environ, clean_env, clear=True):
            apply_yaml_defaults(config)
            assert os.environ.get("S3_BUCKET") == "yaml-bucket"
            assert os.environ.get("S3_REGION") == "eu-west-1"

    def test_does_not_override_existing(self):
        """Existing env vars are preserved (env wins over YAML)."""
        config = {"storage": {"bucket": "yaml-bucket"}}

        with patch.dict(os.environ, {"S3_BUCKET": "env-bucket"}, clear=True):
            apply_yaml_defaults(config)
            assert os.environ["S3_BUCKET"] == "env-bucket"

    def test_bool_coercion(self):
        """Booleans are converted to 'true'/'false' strings."""
        config = {"storage": {"use_ssl": True, "path_style": False}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert os.environ.get("S3_USE_SSL") == "true"
            assert os.environ.get("S3_PATH_STYLE") == "false"

    def test_none_values_skipped(self):
        """None values in YAML do not set env vars."""
        config = {"embeddings": {"vector_size": None}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert "EMBEDDING_VECTOR_SIZE" not in os.environ

    def test_int_coercion(self):
        """Integer values are converted to strings."""
        config = {"storage": {"timeout_seconds": 60}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert os.environ.get("S3_TIMEOUT") == "60"

    def test_empty_string_not_set(self):
        """Empty strings from YAML do not set env vars."""
        config = {"vector_databases": {"qdrant": {"api_key": ""}}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert "QDRANT_API_KEY" not in os.environ


# =============================================================================
# initialize_config Tests
# =============================================================================


class TestInitializeConfig:
    """Tests for initialize_config()."""

    def test_idempotent(self, tmp_path):
        """Safe to call twice - second call is a no-op."""
        import config_loader

        base = {"storage": {"bucket": "idempotent-test"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))

        # Reset the module-level flag
        config_loader._initialized = False

        env_backup = os.environ.copy()
        try:
            os.environ.pop("S3_BUCKET", None)
            os.environ.pop("ENV", None)
            os.environ.pop("PIPELINE_ENV", None)
            initialize_config(config_dir=tmp_path)
            assert os.environ.get("S3_BUCKET") == "idempotent-test"

            # Change YAML - should not affect since already initialized
            base2 = {"storage": {"bucket": "changed-bucket"}}
            (tmp_path / "config.yaml").write_text(yaml.dump(base2))
            os.environ.pop("S3_BUCKET", None)
            initialize_config(config_dir=tmp_path)
            # S3_BUCKET was cleared, but initialize_config won't re-run
            assert "S3_BUCKET" not in os.environ
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
            config_loader._initialized = False

    def test_reads_env_variable(self, tmp_path):
        """Uses ENV env var to select environment overlay."""
        import config_loader

        base = {"storage": {"bucket": "base-bucket"}}
        overlay = {"storage": {"bucket": "dev-bucket"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        (env_dir / "dev.yaml").write_text(yaml.dump(overlay))

        config_loader._initialized = False

        env_backup = os.environ.copy()
        try:
            os.environ.pop("S3_BUCKET", None)
            os.environ["ENV"] = "dev"
            initialize_config(config_dir=tmp_path)
            assert os.environ.get("S3_BUCKET") == "dev-bucket"
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
            config_loader._initialized = False

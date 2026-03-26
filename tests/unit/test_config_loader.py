"""Unit tests for config_loader module.

This file tests the YAML configuration loading, deep merging, environment
variable substitution, and the apply-as-defaults bridging logic.

# Test Coverage

The tests cover:
  - deep_merge: Recursive dict merging, immutability
  - load_yaml_config: Base, environment overlay, secrets loading
  - _resolve_env_substitutions: ${VAR} pattern resolution
  - apply_yaml_defaults: Env var bridging, type coercion, precedence
  - initialize_config: Idempotency, ENV-based overlay selection

# Running Tests

Run with: pytest tests/unit/test_config_loader.py
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
        """Test that flat keys are merged with overlay winning.

        **Why this test is important:**
          - Simplest merge case validates core behavior
          - Overlay values must overwrite base values

        **What it tests:**
          - Overlay key overwrites base key
          - Non-overlapping keys are preserved from both dicts
        """
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test that nested dicts are recursively merged.

        **Why this test is important:**
          - YAML configs are deeply nested (e.g. storage.retry.max_attempts)
          - Must merge at every nesting level without losing sibling keys

        **What it tests:**
          - Nested dict keys are merged recursively
          - Sibling keys at each level are preserved
        """
        base = {"top": {"a": 1, "b": 2}, "other": "x"}
        overlay = {"top": {"b": 3, "c": 4}}
        result = deep_merge(base, overlay)
        assert result == {"top": {"a": 1, "b": 3, "c": 4}, "other": "x"}

    def test_overlay_wins_on_type_conflict(self):
        """Test that overlay replaces base when types conflict.

        **Why this test is important:**
          - An environment overlay may replace a nested section with a scalar
          - Type conflicts must be resolved in favor of the overlay

        **What it tests:**
          - Dict value in base is replaced by string from overlay
        """
        base = {"key": {"nested": True}}
        overlay = {"key": "flat_string"}
        result = deep_merge(base, overlay)
        assert result == {"key": "flat_string"}

    def test_does_not_mutate_inputs(self):
        """Test that neither base nor overlay is mutated.

        **Why this test is important:**
          - Callers may reuse the same base dict across merges
          - Mutation would cause subtle cross-contamination bugs

        **What it tests:**
          - Base dict unchanged after merge
          - Overlay dict unchanged after merge
        """
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        deep_merge(base, overlay)
        assert base == {"a": {"x": 1}}
        assert overlay == {"a": {"y": 2}}

    def test_empty_overlay(self):
        """Test that an empty overlay returns a copy of base.

        **Why this test is important:**
          - Missing environment overlay files produce empty dicts
          - Base config should pass through unchanged

        **What it tests:**
          - Result equals base when overlay is empty
        """
        base = {"a": 1}
        assert deep_merge(base, {}) == {"a": 1}

    def test_empty_base(self):
        """Test that an empty base returns a copy of overlay.

        **Why this test is important:**
          - First merge in a chain starts with empty base
          - Overlay should become the full result

        **What it tests:**
          - Result equals overlay when base is empty
        """
        overlay = {"a": 1}
        assert deep_merge({}, overlay) == {"a": 1}

    def test_deeply_nested(self):
        """Test merging at three or more nesting levels.

        **Why this test is important:**
          - Real configs have paths like ray.circuit_breaker.failure_threshold
          - Must recurse correctly at every level

        **What it tests:**
          - Value at level 3 is overwritten while sibling is preserved
        """
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
        """Test that base config.yaml is loaded when no env is specified.

        **Why this test is important:**
          - Base config is the foundation of all layering
          - Must load and parse correctly without overlays

        **What it tests:**
          - All top-level sections from config.yaml are present
          - Nested values are accessible
        """
        base = {"storage": {"bucket": "test-bucket"}, "embeddings": {"provider": "clip"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))

        result = load_yaml_config(env=None, config_dir=tmp_path)
        assert result["storage"]["bucket"] == "test-bucket"
        assert result["embeddings"]["provider"] == "clip"

    def test_with_env_overlay(self, tmp_path):
        """Test that environment overlay is merged on top of base.

        **Why this test is important:**
          - Environment overlays (dev, staging, prod) customize base config
          - Overlay values must win while base siblings are preserved

        **What it tests:**
          - Overlay value overwrites base value
          - Non-overlapping base values are preserved
        """
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
        """Test that secrets.yaml is merged on top of base and env.

        **Why this test is important:**
          - Secrets (API keys, credentials) are stored separately
          - Must merge after base + env without overwriting non-secret values

        **What it tests:**
          - Secret values are present in merged config
          - Non-secret base values are preserved
        """
        base = {"storage": {"bucket": "default"}}
        secrets = {"storage": {"access_key": "secret123"}}

        (tmp_path / "config.yaml").write_text(yaml.dump(base))
        (tmp_path / "secrets.yaml").write_text(yaml.dump(secrets))

        result = load_yaml_config(env=None, config_dir=tmp_path)
        assert result["storage"]["bucket"] == "default"
        assert result["storage"]["access_key"] == "secret123"

    def test_missing_base_config(self, tmp_path):
        """Test that missing config.yaml returns empty dict.

        **Why this test is important:**
          - Graceful degradation when no config files exist
          - Prevents crashes in unconfigured environments

        **What it tests:**
          - Returns empty dict when config.yaml is absent
        """
        result = load_yaml_config(env=None, config_dir=tmp_path)
        assert result == {}

    def test_missing_env_overlay_is_ok(self, tmp_path):
        """Test that a missing environment overlay is silently skipped.

        **Why this test is important:**
          - Not all environments need an overlay file
          - Should not crash if the overlay is missing

        **What it tests:**
          - Base config returned unchanged when overlay file is absent
        """
        base = {"key": "value"}
        (tmp_path / "config.yaml").write_text(yaml.dump(base))

        result = load_yaml_config(env="nonexistent", config_dir=tmp_path)
        assert result == {"key": "value"}

    def test_env_substitution(self, tmp_path):
        """Test that ${VAR} patterns are resolved from os.environ.

        **Why this test is important:**
          - YAML values may reference environment variables
          - Enables dynamic configuration without duplicating secrets

        **What it tests:**
          - ${VAR_NAME} in YAML value is replaced with env var value
        """
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
        """Test that a single ${VAR} is resolved from os.environ.

        **Why this test is important:**
          - Core substitution mechanism for YAML values
          - Must resolve correctly for single-variable strings

        **What it tests:**
          - ${FOO} is replaced with the value of FOO from os.environ
        """
        with patch.dict(os.environ, {"FOO": "bar"}):
            assert _resolve_env_substitutions("${FOO}") == "bar"

    def test_unresolved_left_as_is(self):
        """Test that unresolved ${VAR} patterns are left unchanged.

        **Why this test is important:**
          - Variables may be set later in the layering process
          - Must not error or blank out unresolved references

        **What it tests:**
          - Unset variable pattern is returned as-is
        """
        result = _resolve_env_substitutions("${DEFINITELY_NOT_SET_12345}")
        assert result == "${DEFINITELY_NOT_SET_12345}"

    def test_multiple_substitutions(self):
        """Test that multiple ${VAR} patterns in one string are all resolved.

        **Why this test is important:**
          - Connection strings often combine host and port variables
          - All patterns in a single string must be resolved

        **What it tests:**
          - Both ${HOST} and ${PORT} are resolved in one string
        """
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            result = _resolve_env_substitutions("http://${HOST}:${PORT}")
            assert result == "http://localhost:8080"

    def test_no_substitution_needed(self):
        """Test that strings without ${VAR} patterns pass through unchanged.

        **Why this test is important:**
          - Most YAML values are plain strings
          - Must not alter strings that don't contain patterns

        **What it tests:**
          - Plain string returned unchanged
        """
        assert _resolve_env_substitutions("plain string") == "plain string"


# =============================================================================
# apply_yaml_defaults Tests
# =============================================================================


class TestApplyYamlDefaults:
    """Tests for apply_yaml_defaults()."""

    def test_sets_unset_vars(self):
        """Test that YAML values are applied when env vars are not set.

        **Why this test is important:**
          - Core purpose of the YAML-as-defaults approach
          - YAML values must fill in missing env vars

        **What it tests:**
          - S3_BUCKET is set from storage.bucket
          - S3_REGION is set from storage.region
        """
        config = {"storage": {"bucket": "yaml-bucket", "region": "eu-west-1"}}

        env_to_clear = ["S3_BUCKET", "S3_REGION"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_to_clear}

        with patch.dict(os.environ, clean_env, clear=True):
            apply_yaml_defaults(config)
            assert os.environ.get("S3_BUCKET") == "yaml-bucket"
            assert os.environ.get("S3_REGION") == "eu-west-1"

    def test_does_not_override_existing(self):
        """Test that existing env vars are preserved over YAML values.

        **Why this test is important:**
          - Env vars always win in the layering order
          - Prevents YAML from overwriting operator-set values

        **What it tests:**
          - Pre-existing S3_BUCKET env var is not overwritten
        """
        config = {"storage": {"bucket": "yaml-bucket"}}

        with patch.dict(os.environ, {"S3_BUCKET": "env-bucket"}, clear=True):
            apply_yaml_defaults(config)
            assert os.environ["S3_BUCKET"] == "env-bucket"

    def test_bool_coercion(self):
        """Test that boolean YAML values are coerced to string.

        **Why this test is important:**
          - os.environ only stores strings
          - Boolean YAML values must be converted to "true"/"false"

        **What it tests:**
          - True becomes "true"
          - False becomes "false"
        """
        config = {"storage": {"use_ssl": True, "path_style": False}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert os.environ.get("S3_USE_SSL") == "true"
            assert os.environ.get("S3_PATH_STYLE") == "false"

    def test_none_values_skipped(self):
        """Test that None values in YAML do not set env vars.

        **Why this test is important:**
          - YAML null/~ values mean "not configured"
          - Must not set env var to the string "None"

        **What it tests:**
          - Env var is absent when YAML value is None
        """
        config = {"embeddings": {"vector_size": None}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert "EMBEDDING_VECTOR_SIZE" not in os.environ

    def test_int_coercion(self):
        """Test that integer YAML values are coerced to string.

        **Why this test is important:**
          - os.environ only stores strings
          - Numeric YAML values must be converted correctly

        **What it tests:**
          - Integer 60 becomes string "60"
        """
        config = {"storage": {"timeout_seconds": 60}}

        with patch.dict(os.environ, {}, clear=True):
            apply_yaml_defaults(config)
            assert os.environ.get("S3_TIMEOUT") == "60"

    def test_empty_string_not_set(self):
        """Test that empty strings from YAML do not set env vars.

        **Why this test is important:**
          - Empty strings in YAML should not override built-in defaults
          - Prevents blanking out values that have code-level defaults

        **What it tests:**
          - Env var is absent when YAML value is empty string
        """
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
        """Test that initialize_config is safe to call multiple times.

        **Why this test is important:**
          - Multiple code paths may call initialize_config()
          - Second call must be a no-op to avoid re-reading changed files

        **What it tests:**
          - First call sets env vars from YAML
          - Second call does not re-process even if YAML changes
        """
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
        """Test that ENV env var selects the correct environment overlay.

        **Why this test is important:**
          - ENV variable drives environment-specific configuration
          - Must load the correct overlay file (e.g. environments/dev.yaml)

        **What it tests:**
          - ENV=dev causes dev.yaml overlay to be merged
          - Overlay value overwrites base value
        """
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

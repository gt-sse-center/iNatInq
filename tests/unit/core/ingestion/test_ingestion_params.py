"""Unit tests for core.services.ingestion_params helpers."""

from __future__ import annotations

import os

from config import EmbeddingConfig, ImageEmbeddingConfig
from core.services.ingestion_params import add_ray_tuning_env, build_image_ingestion_env, build_ingestion_env


def test_build_ingestion_env_includes_required_and_optional(monkeypatch) -> None:
    """Ensure required keys and optional env passthrough are included."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-key")
    monkeypatch.setenv("WEAVIATE_URL", "http://weaviate.test:8080")
    monkeypatch.setenv("WEAVIATE_API_KEY", "weaviate-key")
    monkeypatch.setenv("WEAVIATE_GRPC_HOST", "grpc.weaviate.test")
    monkeypatch.setenv("EXTRA_ENV", "extra-value")

    embedding_config = EmbeddingConfig(
        provider_type="ollama",
        ollama_url="http://ollama.test:11434",
        ollama_model="nomic-embed-text",
        vector_size=768,
    )

    env_vars = build_ingestion_env(
        namespace="ml-system",
        s3_endpoint="http://minio.test:9000",
        s3_access_key_id="access-key",
        s3_secret_access_key="secret-key",
        s3_bucket="bucket",
        s3_prefix="inputs/",
        embedding_config=embedding_config,
        collection="documents",
        extra_env_keys=["EXTRA_ENV"],
    )

    assert env_vars["K8S_NAMESPACE"] == "ml-system"
    assert env_vars["S3_PREFIX"] == "inputs/"
    assert env_vars["S3_ENDPOINT"] == "http://minio.test:9000"
    assert env_vars["S3_ACCESS_KEY_ID"] == "access-key"
    assert env_vars["S3_SECRET_ACCESS_KEY"] == "secret-key"
    assert env_vars["S3_BUCKET"] == "bucket"
    assert env_vars["VECTOR_DB_COLLECTION"] == "documents"
    assert env_vars["EMBEDDING_PROVIDER"] == "ollama"
    assert env_vars["EMBEDDING_VECTOR_SIZE"] == "768"
    assert env_vars["OLLAMA_BASE_URL"] == "http://ollama.test:11434"
    assert env_vars["OLLAMA_MODEL"] == "nomic-embed-text"
    assert env_vars["QDRANT_URL"] == "http://qdrant.test:6333"
    assert env_vars["QDRANT_API_KEY"] == "qdrant-key"
    assert env_vars["WEAVIATE_URL"] == "http://weaviate.test:8080"
    assert env_vars["WEAVIATE_API_KEY"] == "weaviate-key"
    assert env_vars["WEAVIATE_GRPC_HOST"] == "grpc.weaviate.test"
    assert env_vars["EXTRA_ENV"] == "extra-value"


def test_add_ray_tuning_env(monkeypatch) -> None:
    """Ensure RAY_* variables are passed through."""
    monkeypatch.setenv("RAY_NUM_WORKERS", "4")
    monkeypatch.setenv("RAY_BATCH_UPSERT_SIZE", "128")
    monkeypatch.setenv("NOT_RAY", "ignore")

    env_vars: dict[str, str] = {}
    add_ray_tuning_env(env_vars)

    assert env_vars["RAY_NUM_WORKERS"] == "4"
    assert env_vars["RAY_BATCH_UPSERT_SIZE"] == "128"
    assert "NOT_RAY" not in env_vars
    assert os.environ.get("RAY_NUM_WORKERS") == "4"


# ---------------------------------------------------------------------------
# ingestion_targets parameter tests
# ---------------------------------------------------------------------------


def test_build_ingestion_env_with_ingestion_targets(monkeypatch) -> None:
    """Test that VECTOR_DB_TARGETS is set when ingestion_targets is provided.

    **Why this test is important:**
      - Explicit ingestion_targets parameter must be serialized to env var
      - Ray workers read VECTOR_DB_TARGETS to configure single-target mode

    **What it tests:**
      - VECTOR_DB_TARGETS key is present in returned env dict
      - Value matches the provided target
    """
    monkeypatch.delenv("VECTOR_DB_TARGETS", raising=False)
    embedding_config = EmbeddingConfig(provider_type="ollama")

    env_vars = build_ingestion_env(
        namespace="ns",
        s3_endpoint="http://minio:9000",
        s3_access_key_id="ak",
        s3_secret_access_key="sk",
        s3_bucket="b",
        s3_prefix="p/",
        embedding_config=embedding_config,
        collection="col",
        ingestion_targets=frozenset({"qdrant"}),
    )

    assert env_vars["VECTOR_DB_TARGETS"] == "qdrant"


def test_build_ingestion_env_targets_sorted(monkeypatch) -> None:
    """Test that VECTOR_DB_TARGETS value is sorted for deterministic output.

    **Why this test is important:**
      - frozenset iteration order is not guaranteed
      - Sorted output makes logging and debugging predictable

    **What it tests:**
      - VECTOR_DB_TARGETS is "qdrant,weaviate" (alphabetical order)
    """
    monkeypatch.delenv("VECTOR_DB_TARGETS", raising=False)
    embedding_config = EmbeddingConfig(provider_type="ollama")

    env_vars = build_ingestion_env(
        namespace="ns",
        s3_endpoint="http://minio:9000",
        s3_access_key_id="ak",
        s3_secret_access_key="sk",
        s3_bucket="b",
        s3_prefix="p/",
        embedding_config=embedding_config,
        collection="col",
        ingestion_targets=frozenset({"weaviate", "qdrant"}),
    )

    assert env_vars["VECTOR_DB_TARGETS"] == "qdrant,weaviate"


def test_build_ingestion_env_no_targets_omits_key(monkeypatch) -> None:
    """Test that VECTOR_DB_TARGETS is absent when ingestion_targets is None.

    **Why this test is important:**
      - None means "use default" - env var should not be set
      - Prevents overriding the worker's own default behavior

    **What it tests:**
      - VECTOR_DB_TARGETS key is not in returned env dict
    """
    # Ensure env doesn't leak a value in
    monkeypatch.delenv("VECTOR_DB_TARGETS", raising=False)

    embedding_config = EmbeddingConfig(provider_type="ollama")

    env_vars = build_ingestion_env(
        namespace="ns",
        s3_endpoint="http://minio:9000",
        s3_access_key_id="ak",
        s3_secret_access_key="sk",
        s3_bucket="b",
        s3_prefix="p/",
        embedding_config=embedding_config,
        collection="col",
        ingestion_targets=None,
    )

    assert "VECTOR_DB_TARGETS" not in env_vars


def test_build_ingestion_env_targets_from_env_passthrough(monkeypatch) -> None:
    """Test that VECTOR_DB_TARGETS from os.environ is passed through.

    **Why this test is important:**
      - VECTOR_DB_TARGETS is in _VECTOR_ENV_KEYS passthrough list
      - Operators can set it globally and have it forwarded to workers

    **What it tests:**
      - VECTOR_DB_TARGETS from os.environ appears in returned env dict
    """
    monkeypatch.setenv("VECTOR_DB_TARGETS", "weaviate")

    embedding_config = EmbeddingConfig(provider_type="ollama")

    env_vars = build_ingestion_env(
        namespace="ns",
        s3_endpoint="http://minio:9000",
        s3_access_key_id="ak",
        s3_secret_access_key="sk",
        s3_bucket="b",
        s3_prefix="p/",
        embedding_config=embedding_config,
        collection="col",
    )

    # VECTOR_DB_TARGETS is in _VECTOR_ENV_KEYS, so it gets picked up from env
    assert env_vars["VECTOR_DB_TARGETS"] == "weaviate"

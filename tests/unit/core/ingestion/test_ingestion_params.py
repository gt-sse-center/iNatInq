"""Unit tests for core.services.ingestion_params helpers."""

from __future__ import annotations

import os

from config import EmbeddingConfig, ImageEmbeddingConfig
from core.services.ingestion_params import (
    add_ray_tuning_env,
    build_image_ingestion_env,
    build_inat_image_ingestion_env,
    build_ingestion_env,
)


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


def test_build_image_ingestion_env_includes_required_and_optional(monkeypatch) -> None:
    """Ensure image ingestion env includes required keys and passthrough values."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-key")
    monkeypatch.setenv("WEAVIATE_URL", "http://weaviate.test:8080")
    monkeypatch.setenv("WEAVIATE_API_KEY", "weaviate-key")
    monkeypatch.setenv("WEAVIATE_GRPC_HOST", "grpc.weaviate.test")
    monkeypatch.setenv("CLIP_API_KEY", "clip-key")
    monkeypatch.setenv("IMAGE_MAX_ITEMS", "500")
    monkeypatch.setenv("INAT_METADATA_URL", "s3://inaturalist-open-data/photos.csv.gz")
    monkeypatch.setenv("INAT_IMAGE_SIZE", "large")
    monkeypatch.setenv("EXTRA_ENV", "extra-value")

    image_embedding_config = ImageEmbeddingConfig(
        provider_type="clip",
        clip_url="http://clip.test:8000",
        clip_model="ViT-B/32",
        clip_backend="hosted_clip",
        clip_timeout=90,
        clip_circuit_breaker_threshold=7,
        clip_circuit_breaker_timeout=45,
        clip_max_batch_size=16,
        clip_vector_size=512,
        image_batch_size=12,
        image_max_size_mb=8.5,
        image_target_size=336,
    )

    env_vars = build_image_ingestion_env(
        namespace="ml-system",
        s3_endpoint="http://minio.test:9000",
        s3_access_key_id="access-key",
        s3_secret_access_key="secret-key",
        s3_bucket="bucket",
        s3_prefix="images/",
        image_embedding_config=image_embedding_config,
        collection="images",
        extra_env_keys=["EXTRA_ENV"],
    )

    assert env_vars["K8S_NAMESPACE"] == "ml-system"
    assert env_vars["S3_PREFIX"] == "images/"
    assert env_vars["S3_ENDPOINT"] == "http://minio.test:9000"
    assert env_vars["S3_ACCESS_KEY_ID"] == "access-key"
    assert env_vars["S3_SECRET_ACCESS_KEY"] == "secret-key"
    assert env_vars["S3_BUCKET"] == "bucket"
    assert env_vars["VECTOR_DB_COLLECTION"] == "images"
    assert env_vars["IMAGE_EMBEDDING_PROVIDER"] == "clip"
    assert env_vars["CLIP_URL"] == "http://clip.test:8000"
    assert env_vars["CLIP_MODEL"] == "ViT-B/32"
    assert env_vars["CLIP_BACKEND"] == "hosted_clip"
    assert env_vars["CLIP_TIMEOUT"] == "90"
    assert env_vars["CLIP_CIRCUIT_BREAKER_THRESHOLD"] == "7"
    assert env_vars["CLIP_CIRCUIT_BREAKER_TIMEOUT"] == "45"
    assert env_vars["CLIP_MAX_BATCH_SIZE"] == "16"
    assert env_vars["CLIP_VECTOR_SIZE"] == "512"
    assert env_vars["S3_LIST_PAGE_SIZE"] == "12"
    assert env_vars["IMAGE_MAX_SIZE_MB"] == "8.5"
    assert env_vars["IMAGE_TARGET_SIZE"] == "336"
    assert env_vars["QDRANT_URL"] == "http://qdrant.test:6333"
    assert env_vars["QDRANT_API_KEY"] == "qdrant-key"
    assert env_vars["WEAVIATE_URL"] == "http://weaviate.test:8080"
    assert env_vars["WEAVIATE_API_KEY"] == "weaviate-key"
    assert env_vars["WEAVIATE_GRPC_HOST"] == "grpc.weaviate.test"
    assert env_vars["CLIP_API_KEY"] == "clip-key"
    assert env_vars["IMAGE_MAX_ITEMS"] == "500"
    assert env_vars["INAT_METADATA_URL"] == "s3://inaturalist-open-data/photos.csv.gz"
    assert env_vars["INAT_IMAGE_SIZE"] == "large"
    assert env_vars["EXTRA_ENV"] == "extra-value"


def test_build_ingestion_env_with_ingestion_targets(monkeypatch) -> None:
    """Test that VECTOR_DB_TARGETS is set when ingestion_targets is provided."""
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
    """Test that VECTOR_DB_TARGETS value is sorted for deterministic output."""
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

    assert env_vars["K8S_NAMESPACE"] == "ns"
    assert env_vars["VECTOR_DB_TARGETS"] == "qdrant,weaviate"


def test_build_inat_image_ingestion_env_excludes_s3_connection_keys(monkeypatch) -> None:
    """Ensure iNat image ingestion env does not require MinIO/S3 connection values."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("IMAGE_MAX_ITEMS", "500")
    monkeypatch.setenv("INAT_METADATA_URL", "s3://inaturalist-open-data/photos.csv.gz")
    monkeypatch.setenv("EXTRA_ENV", "extra-value")

    image_embedding_config = ImageEmbeddingConfig(
        provider_type="clip",
        clip_url="http://clip.test:8000",
        clip_model="ViT-B/32",
        clip_backend="hosted_clip",
        clip_timeout=90,
        clip_circuit_breaker_threshold=7,
        clip_circuit_breaker_timeout=45,
        clip_max_batch_size=16,
        clip_vector_size=512,
        image_batch_size=12,
        image_max_size_mb=8.5,
        image_target_size=336,
    )

    env_vars = build_inat_image_ingestion_env(
        namespace="ml-system",
        image_embedding_config=image_embedding_config,
        collection="images",
        extra_env_keys=["EXTRA_ENV"],
    )

    assert env_vars["K8S_NAMESPACE"] == "ml-system"
    assert env_vars["VECTOR_DB_COLLECTION"] == "images"
    assert env_vars["IMAGE_EMBEDDING_PROVIDER"] == "clip"
    assert env_vars["IMAGE_MAX_ITEMS"] == "500"
    assert env_vars["INAT_METADATA_URL"] == "s3://inaturalist-open-data/photos.csv.gz"
    assert env_vars["EXTRA_ENV"] == "extra-value"
    assert env_vars["QDRANT_URL"] == "http://qdrant.test:6333"
    assert "S3_ENDPOINT" not in env_vars
    assert "S3_ACCESS_KEY_ID" not in env_vars
    assert "S3_SECRET_ACCESS_KEY" not in env_vars
    assert "S3_BUCKET" not in env_vars


def test_build_inat_image_ingestion_env_passthroughs_vector_db_targets(monkeypatch) -> None:
    """Ensure iNat image ingestion env passes through VECTOR_DB_TARGETS when set."""
    monkeypatch.setenv("VECTOR_DB_TARGETS", "qdrant")

    image_embedding_config = ImageEmbeddingConfig(
        provider_type="clip",
        clip_url="http://clip.test:8000",
        clip_model="ViT-B/32",
    )

    env_vars = build_inat_image_ingestion_env(
        namespace="ml-system",
        image_embedding_config=image_embedding_config,
        collection="images",
    )

    assert env_vars["VECTOR_DB_TARGETS"] == "qdrant"

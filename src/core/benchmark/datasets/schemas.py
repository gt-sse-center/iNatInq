"""Pydantic schemas for validating gold standard dataset JSON files.

This module provides Pydantic v2 models for loading and validating benchmark
datasets from JSON. The schemas enforce structure, required fields, and
value constraints before data reaches the domain layer.

Example:
    ```python
    import json
    from core.benchmark.datasets.schemas import DatasetSchema

    with open("gold-standard.json") as f:
        raw = json.load(f)

    dataset = DatasetSchema.model_validate(raw)
    print(dataset.name, dataset.modality, len(dataset.queries))
    ```
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QuerySchema(BaseModel):
    """Schema for a single benchmark query.

    Attributes:
        id: Unique query identifier.
        text: Query text (e.g., "red bird").
        relevant: List of relevant document IDs for binary relevance.
        graded_relevance: Optional mapping of document IDs to integer
            relevance grades. Higher grades indicate more relevant documents.

    Example:
        ```json
        {
            "id": "q1",
            "text": "red bird",
            "relevant": ["img1", "img3"],
            "graded_relevance": {"img1": 3, "img3": 1}
        }
        ```
    """

    id: str
    text: str
    relevant: list[str] = Field(min_length=1)
    graded_relevance: dict[str, int] | None = None

    @model_validator(mode="after")
    def graded_keys_must_be_subset_of_relevant(self) -> QuerySchema:
        """Validate that graded_relevance keys are a subset of relevant IDs."""
        if self.graded_relevance is not None:
            relevant_set = set(self.relevant)
            extra = set(self.graded_relevance) - relevant_set
            if extra:
                msg = f"graded_relevance contains IDs not in relevant: {sorted(extra)}"
                raise ValueError(msg)
        return self


class MetadataSchema(BaseModel):
    """Schema for optional dataset metadata.

    All fields are optional. This captures provenance information about
    how the gold standard dataset was created.

    Attributes:
        created_at: ISO 8601 timestamp of dataset creation.
        annotators: List of annotator identifiers.
        agreement_threshold: Inter-annotator agreement threshold used.

    Example:
        ```json
        {
            "created_at": "2026-02-09T12:00:00Z",
            "annotators": ["human-1", "human-2"],
            "agreement_threshold": 0.8
        }
        ```
    """

    created_at: str | None = None
    annotators: list[str] | None = None
    agreement_threshold: float | None = None


class DatasetSchema(BaseModel):
    """Schema for a complete benchmark dataset.

    Validates the top-level structure of a gold standard JSON file
    including name, modality, queries, and optional metadata.

    Attributes:
        name: Unique dataset identifier.
        description: Human-readable description of the dataset.
        modality: Data modality, must be "text" or "image".
        queries: Non-empty list of query schemas.
        metadata: Optional provenance metadata.

    Example:
        ```json
        {
            "name": "sample-gold-standard",
            "description": "Synthetic image benchmark",
            "modality": "image",
            "queries": [
                {"id": "q1", "text": "red bird", "relevant": ["img1"]}
            ]
        }
        ```
    """

    name: str
    description: str
    modality: Literal["text", "image"]
    queries: list[QuerySchema] = Field(min_length=1)
    metadata: MetadataSchema | None = None

    @model_validator(mode="after")
    def query_ids_must_be_unique(self) -> DatasetSchema:
        """Validate that all query IDs are unique."""
        ids = [q.id for q in self.queries]
        duplicates = {qid for qid in ids if ids.count(qid) > 1}
        if duplicates:
            msg = f"Duplicate query IDs: {sorted(duplicates)}"
            raise ValueError(msg)
        return self

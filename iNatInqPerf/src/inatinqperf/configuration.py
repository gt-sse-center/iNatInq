"""Modules with classes for loading configurations with Pydantic validation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, PositiveInt, StringConstraints, field_validator
from simpleeval import simple_eval

from inatinqperf.adaptors.enums import Metric

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding."""

    model_id: NonEmptyStr


class VectorDatabaseConfig(BaseModel):
    """Configuration for Vector Database."""

    type: NonEmptyStr


class SearchParams(BaseModel):
    """Configuration for search parameters."""

    topk: int
    queries_file: Path
    limit: int = -1


class BaselineResults(BaseModel):
    """Configuration for holding paths for search results on baseline vector database.

    Paths should be specified with respect to the iNatInqPerf root directory.
    Used for computing recall.
    """

    results: Path
    results_post_update: Path

    @field_validator("results", "results_post_update", mode="after")
    @classmethod
    def get_absolute_path(cls, value: Path) -> Path:
        """Get the absolute path for the results."""
        return Path(__file__).parent.parent.parent / value


class Config(BaseModel):
    """Class encapsulating benchmark configuration with data validation."""

    search: SearchParams
    compute_recall: bool = False
    baseline: BaselineResults
    embedding_model: EmbeddingModelConfig

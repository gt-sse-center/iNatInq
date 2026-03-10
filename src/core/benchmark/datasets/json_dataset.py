"""JSON file loader for gold standard benchmark datasets.

This module implements the Dataset interface for loading benchmark datasets
from JSON files, with Pydantic validation on load.

Example:
    ```python
    from pathlib import Path
    from core.benchmark.datasets.json_dataset import JSONDataset

    dataset = JSONDataset.from_file(Path("gold-standard.json"))
    print(dataset.name, dataset.modality, len(dataset))

    for query in dataset.queries():
        print(query.id, query.text, query.relevant)
    ```
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from core.benchmark.datasets.base import Dataset, Query
from core.benchmark.datasets.schemas import DatasetSchema
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class JSONDataset(Dataset):
    """Dataset loaded from a JSON file with Pydantic validation.

    Validates the JSON structure on load using DatasetSchema, then
    converts to domain Query objects and caches them in memory.

    Attributes:
        _name: Dataset name from the JSON file.
        _modality: Data modality ("text" or "image").
        _queries: Cached list of Query objects.
    """

    def __init__(
        self,
        *,
        name: str,
        modality: Literal["text", "image"],
        queries: list[Query],
    ) -> None:
        self._name = name
        self._modality = modality
        self._queries = queries

    @property
    @override
    def name(self) -> str:
        """Return the dataset name."""
        return self._name

    @property
    @override
    def modality(self) -> Literal["text", "image"]:
        """Return the data modality ("text" or "image")."""
        return self._modality

    @override
    def queries(self) -> Iterator[Query]:
        """Iterate over queries in the dataset.

        Returns:
            Iterator of Query objects.
        """
        return iter(self._queries)

    @override
    def __len__(self) -> int:
        """Return the number of queries in the dataset."""
        return len(self._queries)

    @classmethod
    def from_file(cls, path: Path) -> JSONDataset:
        """Load and validate a dataset from a JSON file.

        Reads the JSON file, validates it against DatasetSchema, then
        converts to domain Query objects.

        Args:
            path: Path to the JSON dataset file.

        Returns:
            A JSONDataset instance with cached queries.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            pydantic.ValidationError: If the JSON does not match the schema.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        schema = DatasetSchema.model_validate(raw)

        queries = [
            Query(
                id=q.id,
                text=q.text,
                relevant=set(q.relevant),
                graded_relevance=q.graded_relevance or {},
            )
            for q in schema.queries
        ]

        return cls(name=schema.name, modality=schema.modality, queries=queries)

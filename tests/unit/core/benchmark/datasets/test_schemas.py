"""Unit tests for benchmark dataset Pydantic schemas.

Tests validation rules for benchmark dataset JSON schemas (query/document IDs,
relevance scores, required fields). This is important because schema validation
catches malformed dataset files before they corrupt benchmark results.
"""

import pytest
from pydantic import ValidationError

from core.benchmark.datasets.schemas import DatasetSchema, MetadataSchema, QuerySchema


class TestQuerySchema:
    """Tests for QuerySchema validation."""

    def test_valid_query(self):
        """Valid query with required fields."""
        q = QuerySchema(id="q1", text="red bird", relevant=["img1", "img3"])

        assert q.id == "q1"
        assert q.text == "red bird"
        assert q.relevant == ["img1", "img3"]
        assert q.graded_relevance is None

    def test_valid_query_with_graded_relevance(self):
        """Valid query with optional graded_relevance."""
        q = QuerySchema(
            id="q1",
            text="red bird",
            relevant=["img1", "img3"],
            graded_relevance={"img1": 3, "img3": 1},
        )

        assert q.graded_relevance == {"img1": 3, "img3": 1}

    def test_missing_id_raises(self):
        """Missing id field raises ValidationError."""
        with pytest.raises(ValidationError):
            QuerySchema(text="red bird", relevant=["img1"])

    def test_missing_text_raises(self):
        """Missing text field raises ValidationError."""
        with pytest.raises(ValidationError):
            QuerySchema(id="q1", relevant=["img1"])

    def test_missing_relevant_raises(self):
        """Missing relevant field raises ValidationError."""
        with pytest.raises(ValidationError):
            QuerySchema(id="q1", text="red bird")

    def test_empty_relevant_raises(self):
        """Empty relevant list raises ValidationError (min_length=1)."""
        with pytest.raises(ValidationError, match="too_short"):
            QuerySchema(id="q1", text="red bird", relevant=[])

    def test_graded_keys_not_in_relevant_raises(self):
        """graded_relevance with IDs not in relevant raises ValidationError."""
        with pytest.raises(ValidationError, match="not in relevant"):
            QuerySchema(
                id="q1",
                text="red bird",
                relevant=["img1"],
                graded_relevance={"img1": 3, "img99": 1},
            )

    def test_graded_subset_of_relevant_passes(self):
        """graded_relevance as a subset of relevant is valid."""
        q = QuerySchema(
            id="q1",
            text="red bird",
            relevant=["img1", "img3", "img5"],
            graded_relevance={"img1": 3},
        )

        assert q.graded_relevance == {"img1": 3}


class TestMetadataSchema:
    """Tests for MetadataSchema validation."""

    def test_all_fields_optional(self):
        """All metadata fields are optional."""
        m = MetadataSchema()

        assert m.created_at is None
        assert m.annotators is None
        assert m.agreement_threshold is None

    def test_with_all_fields(self):
        """Metadata with all fields populated."""
        m = MetadataSchema(
            created_at="2026-02-09T12:00:00Z",
            annotators=["human-1", "human-2"],
            agreement_threshold=0.8,
        )

        assert m.created_at == "2026-02-09T12:00:00Z"
        assert m.annotators == ["human-1", "human-2"]
        assert m.agreement_threshold == 0.8

    def test_partial_metadata(self):
        """Metadata with only some fields set."""
        m = MetadataSchema(annotators=["annotator-1"])

        assert m.annotators == ["annotator-1"]
        assert m.created_at is None
        assert m.agreement_threshold is None


class TestDatasetSchema:
    """Tests for DatasetSchema validation."""

    def _minimal_dataset(self, **overrides) -> dict:
        """Helper to build a minimal valid dataset dict."""
        base = {
            "name": "test-dataset",
            "description": "A test dataset",
            "modality": "text",
            "queries": [
                {"id": "q1", "text": "query one", "relevant": ["d1"]},
            ],
        }
        base.update(overrides)
        return base

    def test_valid_dataset(self):
        """Valid dataset with required fields."""
        ds = DatasetSchema.model_validate(self._minimal_dataset())

        assert ds.name == "test-dataset"
        assert ds.description == "A test dataset"
        assert ds.modality == "text"
        assert len(ds.queries) == 1
        assert ds.metadata is None

    def test_valid_dataset_image_modality(self):
        """Dataset with image modality."""
        ds = DatasetSchema.model_validate(self._minimal_dataset(modality="image"))

        assert ds.modality == "image"

    def test_invalid_modality_raises(self):
        """Invalid modality raises ValidationError."""
        with pytest.raises(ValidationError):
            DatasetSchema.model_validate(self._minimal_dataset(modality="audio"))

    def test_missing_name_raises(self):
        """Missing name raises ValidationError."""
        data = self._minimal_dataset()
        del data["name"]

        with pytest.raises(ValidationError):
            DatasetSchema.model_validate(data)

    def test_missing_description_raises(self):
        """Missing description raises ValidationError."""
        data = self._minimal_dataset()
        del data["description"]

        with pytest.raises(ValidationError):
            DatasetSchema.model_validate(data)

    def test_missing_queries_raises(self):
        """Missing queries raises ValidationError."""
        data = self._minimal_dataset()
        del data["queries"]

        with pytest.raises(ValidationError):
            DatasetSchema.model_validate(data)

    def test_empty_queries_raises(self):
        """Empty queries list raises ValidationError."""
        with pytest.raises(ValidationError, match="too_short"):
            DatasetSchema.model_validate(self._minimal_dataset(queries=[]))

    def test_duplicate_query_ids_raises(self):
        """Duplicate query IDs raise ValidationError."""
        queries = [
            {"id": "q1", "text": "first", "relevant": ["d1"]},
            {"id": "q1", "text": "duplicate", "relevant": ["d2"]},
        ]

        with pytest.raises(ValidationError, match="Duplicate query IDs"):
            DatasetSchema.model_validate(self._minimal_dataset(queries=queries))

    def test_with_metadata(self):
        """Dataset with metadata block."""
        data = self._minimal_dataset(
            metadata={
                "created_at": "2026-02-09T12:00:00Z",
                "annotators": ["human-1"],
                "agreement_threshold": 0.9,
            }
        )
        ds = DatasetSchema.model_validate(data)

        assert ds.metadata is not None
        assert ds.metadata.created_at == "2026-02-09T12:00:00Z"
        assert ds.metadata.annotators == ["human-1"]
        assert ds.metadata.agreement_threshold == 0.9

    def test_with_empty_metadata(self):
        """Dataset with empty metadata object."""
        ds = DatasetSchema.model_validate(self._minimal_dataset(metadata={}))

        assert ds.metadata is not None
        assert ds.metadata.created_at is None

    def test_multiple_queries_with_graded(self):
        """Dataset with multiple queries, some with graded relevance."""
        queries = [
            {
                "id": "q1",
                "text": "red bird",
                "relevant": ["img1", "img3"],
                "graded_relevance": {"img1": 3, "img3": 1},
            },
            {
                "id": "q2",
                "text": "blue fish",
                "relevant": ["img2"],
            },
        ]
        ds = DatasetSchema.model_validate(self._minimal_dataset(queries=queries))

        assert len(ds.queries) == 2
        assert ds.queries[0].graded_relevance == {"img1": 3, "img3": 1}
        assert ds.queries[1].graded_relevance is None

    def test_invalid_query_in_list_raises(self):
        """Invalid query inside queries list raises ValidationError."""
        queries = [
            {"id": "q1", "text": "valid", "relevant": ["d1"]},
            {"id": "q2", "text": "missing relevant"},
        ]

        with pytest.raises(ValidationError):
            DatasetSchema.model_validate(self._minimal_dataset(queries=queries))

    def test_model_validate_from_json_dict(self):
        """model_validate works with a plain dict (simulating json.load)."""
        raw = {
            "name": "gold-standard",
            "description": "Benchmark dataset",
            "modality": "image",
            "queries": [
                {"id": "q1", "text": "red circle", "relevant": ["img1", "img2"]},
                {
                    "id": "q2",
                    "text": "blue square",
                    "relevant": ["img3"],
                    "graded_relevance": {"img3": 2},
                },
            ],
            "metadata": {"created_at": "2026-01-01T00:00:00Z"},
        }

        ds = DatasetSchema.model_validate(raw)

        assert ds.name == "gold-standard"
        assert ds.modality == "image"
        assert len(ds.queries) == 2
        assert ds.metadata.created_at == "2026-01-01T00:00:00Z"

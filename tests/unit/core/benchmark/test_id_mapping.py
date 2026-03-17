"""Unit tests for benchmark ID mapping module."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import pytest

from core.benchmark.id_mapping import S3KeyIDMapper


class TestS3KeyIDMapper:
    """Tests for S3KeyIDMapper."""

    def test_doc_id_to_point_id_deterministic(self):
        """UUID5 derivation is deterministic for the same doc_id."""
        mapper = S3KeyIDMapper()
        result1 = mapper.doc_id_to_point_id("1316621")
        result2 = mapper.doc_id_to_point_id("1316621")
        assert result1 == result2

    def test_doc_id_to_point_id_matches_uuid5(self):
        """doc_id_to_point_id matches uuid5(NAMESPACE_URL, doc_id)."""
        mapper = S3KeyIDMapper()
        doc_id = "1316621"
        expected = str(uuid5(NAMESPACE_URL, doc_id))
        assert mapper.doc_id_to_point_id(doc_id) == expected

    def test_doc_id_to_point_id_different_ids_differ(self):
        """Different doc IDs produce different point IDs."""
        mapper = S3KeyIDMapper()
        assert mapper.doc_id_to_point_id("1") != mapper.doc_id_to_point_id("2")

    def test_point_id_to_doc_id_from_payload(self):
        """point_id_to_doc_id reads s3_key from payload."""
        mapper = S3KeyIDMapper()
        payload = {"s3_key": "1316621", "other": "data"}
        result = mapper.point_id_to_doc_id("some-uuid", payload)
        assert result == "1316621"

    def test_point_id_to_doc_id_strips_prefix_from_s3_key(self):
        """point_id_to_doc_id extracts basename when s3_key has a path prefix."""
        mapper = S3KeyIDMapper()
        payload = {"s3_key": "benchmark-images/1009530"}
        result = mapper.point_id_to_doc_id("some-uuid", payload)
        assert result == "1009530"

    def test_point_id_to_doc_id_strips_nested_prefix(self):
        """point_id_to_doc_id handles deeply nested s3_key paths."""
        mapper = S3KeyIDMapper()
        payload = {"s3_key": "a/b/c/42"}
        result = mapper.point_id_to_doc_id("some-uuid", payload)
        assert result == "42"

    def test_point_id_to_doc_id_missing_s3_key_falls_back(self):
        """Falls back to point_id when s3_key not in payload."""
        mapper = S3KeyIDMapper()
        payload = {"other": "data"}
        result = mapper.point_id_to_doc_id("some-uuid", payload)
        assert result == "some-uuid"

    def test_point_id_to_doc_id_empty_payload_falls_back(self):
        """Falls back to point_id when payload is empty."""
        mapper = S3KeyIDMapper()
        result = mapper.point_id_to_doc_id("some-uuid", {})
        assert result == "some-uuid"

    def test_roundtrip_doc_id(self):
        """doc_id → point_id → payload lookup → doc_id roundtrip works."""
        mapper = S3KeyIDMapper()
        doc_id = "42"
        point_id = mapper.doc_id_to_point_id(doc_id)
        payload = {"s3_key": doc_id}
        assert mapper.point_id_to_doc_id(point_id, payload) == doc_id

    def test_batch_doc_ids_to_point_ids(self):
        """batch_doc_ids_to_point_ids converts multiple IDs."""
        mapper = S3KeyIDMapper()
        doc_ids = ["1", "2", "3"]
        result = mapper.batch_doc_ids_to_point_ids(doc_ids)
        assert len(result) == 3
        for doc_id, point_id in zip(doc_ids, result):
            assert point_id == str(uuid5(NAMESPACE_URL, doc_id))

    def test_batch_doc_ids_to_point_ids_empty(self):
        """batch_doc_ids_to_point_ids handles empty list."""
        mapper = S3KeyIDMapper()
        assert mapper.batch_doc_ids_to_point_ids([]) == []

"""ID mapping between INQUIRE document IDs and Qdrant point IDs.

Qdrant points are indexed using UUID5 derived from the iNat photo ID:
``uuid5(NAMESPACE_URL, photo_id)``. The payload ``s3_key`` field stores
the original numeric photo ID for reverse mapping.

Example:
    ```python
    from core.benchmark.id_mapping import S3KeyIDMapper

    mapper = S3KeyIDMapper()
    point_id = mapper.doc_id_to_point_id("1316621")
    # "000019de-..."

    doc_id = mapper.point_id_to_doc_id(point_id, {"s3_key": "1316621"})
    # "1316621"
    ```
"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import attrs


@attrs.define(frozen=True, slots=True)
class S3KeyIDMapper:
    """Maps between document IDs (iNat photo IDs) and Qdrant point IDs (UUID5).

    The mapping convention is:
    - Forward: ``str(uuid5(NAMESPACE_URL, doc_id))``
    - Reverse: read ``payload["s3_key"]`` from the Qdrant point payload
    """

    def doc_id_to_point_id(self, doc_id: str) -> str:
        """Convert a document ID to a Qdrant point ID.

        Args:
            doc_id: Plain numeric photo ID (e.g., ``"1316621"``).

        Returns:
            UUID5 string derived from the doc ID.
        """
        return str(uuid5(NAMESPACE_URL, doc_id))

    def point_id_to_doc_id(self, point_id: str, payload: dict[str, object]) -> str:
        """Extract the document ID from a Qdrant point's payload.

        Reads the ``s3_key`` field from the payload. Falls back to the
        raw ``point_id`` if ``s3_key`` is not present.

        Args:
            point_id: UUID string of the Qdrant point.
            payload: Payload dictionary from the Qdrant point.

        Returns:
            The original document ID.
        """
        s3_key = payload.get("s3_key")
        if isinstance(s3_key, str):
            # Extract basename — s3_key may include a prefix path
            # (e.g., "benchmark-images/1009530" → "1009530")
            return s3_key.rsplit("/", 1)[-1]
        return point_id

    def batch_doc_ids_to_point_ids(self, doc_ids: list[str]) -> list[str]:
        """Convert a batch of document IDs to Qdrant point IDs.

        Args:
            doc_ids: List of document IDs.

        Returns:
            List of UUID5 strings in the same order.
        """
        return [self.doc_id_to_point_id(doc_id) for doc_id in doc_ids]

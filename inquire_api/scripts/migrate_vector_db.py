"""Helper script to migrate vector database from one collection to another.

NOTE: Specifically written for Qdrant.
"""

from qdrant_client import QdrantClient

source_client = QdrantClient(
    url="localhost",
    port=7433,
    grpc_port=7434,
    prefer_grpc=True,
    timeout=10,  # extend the timeout to 10 seconds
)

target_client = QdrantClient(
    url="localhost",
    port=7333,
    grpc_port=7334,
    prefer_grpc=True,
    timeout=10,  # extend the timeout to 10 seconds
)


collection_name = "inatinq"

dataset_size = source_client.count(collection_name=collection_name).count
print(f"Migrating {dataset_size} data points")
source_client.migrate(
    target_client,
    [collection_name],
    batch_size=2048,
    recreate_on_collision=True,
)

"""Unit tests for the vector database adaptor."""

import pytest

from inquire_api.vector_db import VectorDatabaseAdaptor


@pytest.fixture(name="bare_vector_db")
def bare_vector_db_fixture(port, grpc_port):
    """Create client without any data."""
    collection_name = "test"
    vector_db = VectorDatabaseAdaptor(collection_name=collection_name, port=port, grpc_port=grpc_port)

    yield vector_db

    # Clean up the collection
    vector_db.client.delete_collection(collection_name=collection_name)


def test_init():
    """Test constructor."""
    vector_db = VectorDatabaseAdaptor(collection_name="test")
    assert vector_db.collection_name == "test"
    assert vector_db.metric == "cosine"


def test_initialize_collection(bare_vector_db, dataset, N):
    bare_vector_db.initialize_collection(dataset, batch_size=64)

    num_points_in_db = bare_vector_db.client.count(
        collection_name=bare_vector_db.collection_name,
        exact=True,
    ).count
    assert num_points_in_db == N


def test_stats(vector_db):
    stats = vector_db.stats()
    assert stats["metric"] == "cosine"
    assert stats["m"] == 32
    assert stats["ef"] == 128

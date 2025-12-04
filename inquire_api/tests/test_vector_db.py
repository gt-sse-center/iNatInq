"""Unit tests for the vector database adaptor."""

import numpy as np
from datasets import Dataset, Features, Value, List
import pytest

from datetime import datetime
from inquire_api.vector_db import VectorDatabaseAdaptor


@pytest.fixture(name="N")
def dataset_size_fixture():
    return 10


@pytest.fixture(name="dataset")
def dataset_fixture(N):
    rng = np.random.default_rng(42)

    dataset = Dataset.from_dict(
        {
            "id": np.arange(N).tolist(),
            "img_url": [f"https://test.com/{i}.png" for i in range(N)],
            "file_name": [f"image_{i}.png" for i in range(N)],
            "img_embedding": [rng.random(512) for _ in range(N)],
            "latitude": rng.random(size=(N,)).tolist(),
            "longitude": rng.random(size=(N,)).tolist(),
            "positional_accuracy": rng.random(size=(N,)).tolist(),
            "observed_on": [datetime.now() for _ in range(N)],
            "taxon": ["" for _ in range(N)],
        },
        features=Features(
            {
                # `id` column to be of type int64
                "id": Value("int64"),
                "img_url": Value("string"),
                "file_name": Value("string"),
                # `img_embedding` column is of type datasets.List[float32]
                "img_embedding": List(feature=Value("float32"), length=512),
                "latitude": Value("float32"),
                "longitude": Value("float32"),
                "positional_accuracy": Value("float32"),
                "observed_on": Value("date32"),
                "taxon": Value("string"),
            },
        ),
    )
    return dataset


@pytest.fixture(name="vector_db")
def vector_db_fixture():
    # Create client without any data
    collection_name = "test"
    vector_db = VectorDatabaseAdaptor(collection_name=collection_name)

    yield vector_db

    # Clean up the collection
    vector_db.client.delete_collection(collection_name=collection_name)


def test_init():
    """Test constructor."""
    vector_db = VectorDatabaseAdaptor(collection_name="test")
    assert vector_db.collection_name == "test"
    assert vector_db.metric == "cosine"


def test_initialize_collection(vector_db, dataset, N):
    vector_db.initialize_collection(dataset, batch_size=64)

    num_points_in_db = vector_db.client.count(
        collection_name=vector_db.collection_name,
        exact=True,
    ).count
    assert num_points_in_db == N


def test_stats(vector_db):
    stats = vector_db.stats()
    assert stats["metric"] == "cosine"
    assert stats["m"] == 32
    assert stats["ef"] == 128

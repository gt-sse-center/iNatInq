"""Common pytest configurations and fixtures."""

from datetime import datetime

import numpy as np
import pytest
from datasets import Dataset, Features, List, Value

from inquire_api.container import Container, ContainerConfig
from inquire_api.vector_db import VectorDatabaseAdaptor


@pytest.fixture(name="port", scope="session")
def vector_db_port_fixture():
    return 9333


@pytest.fixture(name="grpc_port", scope="session")
def vector_db_grpc_port_fixture():
    return 9334


@pytest.fixture(autouse=True, scope="session")
def container_fixture(port, grpc_port):
    cfg = ContainerConfig(name="vdb-test-container", ports={6333: port, 6334: grpc_port})
    container = Container(container_cfg=cfg, remove_on_stop=True, auto_stop=True)

    yield container

    del container


@pytest.fixture(name="N", scope="session")
def dataset_size_fixture():
    return 256


@pytest.fixture(name="dataset", scope="session")
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
            "observed_on": [datetime.now(tz=datetime.UTC) for _ in range(N)],
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
def vector_db_fixture(dataset, port, grpc_port):
    """Populate vector database for testing."""
    collection_name = "test"
    vector_db = VectorDatabaseAdaptor(collection_name=collection_name, port=port, grpc_port=grpc_port)

    vector_db.initialize_collection(dataset, batch_size=64)

    yield vector_db

    # Clean up the collection
    vector_db.client.delete_collection(collection_name=collection_name)

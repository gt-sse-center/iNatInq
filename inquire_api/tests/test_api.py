"""Unit tests for the API."""

import pytest
from fastapi.testclient import TestClient

from inquire_api.config import Settings
from inquire_api.main import app, get_settings


@pytest.fixture(name="client")
def client_fixture(vector_db, port, grpc_port):
    app.dependency_overrides[get_settings] = lambda: Settings(
        vectordb_collection_name=vector_db.collection_name, vectordb_port=port, vectordb_grpc_port=grpc_port
    )

    with TestClient(app) as client:
        yield client


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"app": "The Inquire API"}


def test_count(client):
    response = client.get("/count")
    assert response.json() == 256


def test_query(client):
    form_data = {"user_input": "a sleeping dog", "k": 10, "filters": "{}"}
    response = client.post("/query", data=form_data)

    results = response.json()
    assert len(results) == 10

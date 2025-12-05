"""Unit tests for the API."""

import os
import pytest
from fastapi.testclient import TestClient

from inquire_api.main import app

# TODO(Varun): Make fixtures
pytestmark = pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Running on Github Actions CI")


@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"app": "The Inquire API"}


def test_count(client):
    response = client.get("/count")
    assert response.json() == 2091851


def test_query(client):
    form_data = {"user_input": "a sleeping dog", "k": 10, "filters": "{}"}
    response = client.post("/query", data=form_data)

    results = response.json()
    assert len(results) == 10

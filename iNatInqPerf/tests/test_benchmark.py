"""Tests for the benchmarking code."""

import numpy as np
import pytest
from collections.abc import Sequence
from datasets import Dataset as HuggingFaceDataset

from inatinqperf import adaptors
from inatinqperf.adaptors.base import SearchResult
from inatinqperf.adaptors.enums import Metric
from inatinqperf.benchmark import Benchmarker, benchmark
from inatinqperf.configuration import VectorDatabaseConfig


@pytest.fixture(name="data_path", scope="session")
def data_path_fixture(tmp_path_factory):
    """Fixture to return a temporary data path which can be used for all tests within a session.

    The common path will ensure the HuggingFace dataset isn't repeatedly downloaded.
    """
    return tmp_path_factory.mktemp("data")



@pytest.fixture(name="benchmark_module")
def mocked_benchmark_module(monkeypatch):
    def _fake_ds_embeddings(path=None, splits=None):
        n = 256
        d = 64
        rng = np.random.default_rng(42)
        data_dict = {
            "id": list(range(n)),
            "embedding": [rng.uniform(0, 100, d).astype(np.float32) for _ in range(n)],
        }

        return HuggingFaceDataset.from_dict(data_dict)

    # patch benchmark.load_huggingface_dataset
    monkeypatch.setattr(benchmark, "load_huggingface_dataset", _fake_ds_embeddings)
    return benchmark


class MockExactBaseline:
    """A mock of an exact baseline index such as FAISS Flat."""

    def search(self, q, k) -> Sequence[SearchResult]:
        ids = np.arange(k)
        scores = np.zeros_like(ids, dtype=np.float32)
        return [SearchResult(id=i, score=score) for i, score in zip(ids, scores)]


def test_load_cfg(config_yaml, data_path):
    benchmarker = Benchmarker(config_yaml, base_path=data_path)

    assert benchmarker.cfg.embedding_model.model_id == "openai/clip-vit-base-patch32"
    assert benchmarker.cfg.vectordb.type == "qdrant"
    assert benchmarker.cfg.search.topk == 10
    assert benchmarker.cfg.search.queries_file == "benchmark/queries.txt"
    assert benchmarker.cfg.baseline.results == "tests/fixtures/baseline_results.npy"
    assert benchmarker.cfg.baseline.results_post_update == "tests/fixtures/baseline_results_post_update.npy"

    # Bad path: missing file raises (FileNotFoundError or OSError depending on impl)
    with pytest.raises((FileNotFoundError, OSError, IOError)):
        Benchmarker(data_path / "nope.yaml", base_path=data_path)


def test_search(config_yaml, data_path, caplog):
    """Test vector DB search."""
    benchmarker = Benchmarker(config_yaml, base_path=data_path)

    # TODO: update this test after search functionality is updated to call the FastAPI search route

    assert True


# ---------- Edge cases for helpers ----------
def test_recall_at_k_edges():
    # No hits when there are no neighbors (1 row, 0 columns -> denominator = 1*k)
    I_true = np.empty((1, 0), dtype=int)
    I_test = np.empty((1, 0), dtype=int)
    assert benchmark.recall_at_k(I_true, I_test, 1) == 0.0

    # k larger than available neighbors
    I_true = np.array([[0]], dtype=int)
    I_test = np.array([[0, 1, 2]], dtype=int)
    assert 0.0 <= benchmark.recall_at_k(I_true, I_test, 5) <= 1.0


def test_run_all(config_yaml, tmp_path, caplog):
    benchmarker = Benchmarker(config_yaml, base_path=tmp_path)
    # Disable distributed deployment knobs for unit tests to keep FAISS setup deterministic.
    benchmarker.cfg.vectordb.params.index_type = "FLAT"
    benchmarker.cfg.containers = []
    benchmarker.cfg.container_network = ""
    benchmarker.run()

    # Mirror the log assertion with whatever index type the config specifies.
    expected_index_type = benchmarker.cfg.vectordb.params.index_type.upper()
    assert expected_index_type in caplog.text
    assert "topk" in caplog.text and "10" in caplog.text

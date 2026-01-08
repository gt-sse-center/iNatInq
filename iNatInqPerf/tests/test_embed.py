"""Tests for the `embed` module."""

import numpy as np
import pytest
from PIL import Image

from datasets import Dataset as HFDataset
from inatinqperf.utils import embed


@pytest.fixture(autouse=True)
def clip_model_fixture(mocker):
    class DummyModel:
        """Dummy embedding model to use as a mock."""

        def __init__(self, model_id: str):
            self.model_id = model_id
            self.embedding_dim: int = 512

        def __call__(
            self,
            images: list | None = None,
            text: list | None = None,
        ) -> np.ndarray:
            if images:
                return np.zeros((len(images), self.embedding_dim))
            if text:
                return np.zeros((len(text), self.embedding_dim))

    mocker.patch("inatinqperf.utils.embed.PretrainedCLIPModel", DummyModel)


# -----------------------
# Tests
# -----------------------


def test_embed_text():
    X = embed.embed_text(["hello", "world"], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 2


def test_embed_text_empty():
    X = embed.embed_text([], "dummy-model")
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 0

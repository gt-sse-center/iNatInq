"""Unit tests for the embed module."""

import pytest

from inat_toolkit.embed import ImageEmbedder, TextEmbedder


@pytest.fixture(name="model_id")
def model_id_fixture():
    return "openai/clip-vit-base-patch16"


@pytest.fixture(name="image_embedder")
def image_embedder_fixture(model_id):
    return ImageEmbedder(model_id=model_id)


@pytest.fixture(name="text_embedder")
def text_embedder_fixture(model_id):
    return TextEmbedder(model_id=model_id)


def test_image_embedder_init(model_id):
    embedder = ImageEmbedder(model_id)
    assert isinstance(embedder, ImageEmbedder)


def test_text_embedding(text_embedder):
    text = "this is a test"
    feats = text_embedder(text)
    assert feats.shape == (1, text_embedder.embedding_dim)


def test_text_list_embedding(text_embedder):
    text = ["this is a query", "sleepy dog", "let's embed!"]
    feats = text_embedder(text)
    assert feats.shape == (len(text), text_embedder.embedding_dim)

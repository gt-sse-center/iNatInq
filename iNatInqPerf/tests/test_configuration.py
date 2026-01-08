from pathlib import Path

from inatinqperf.configuration import (
    Config,
    EmbeddingModelConfig,
    BaselineResults,
    SearchParams,
    VectorDatabaseConfig,
)


def test_embedding_params(benchmark_yaml):
    params = EmbeddingModelConfig(**benchmark_yaml["embedding_model"])
    assert params.model_id == "openai/clip-vit-base-patch32"


def test_vectordatabase_params(benchmark_yaml):
    params = VectorDatabaseConfig(**benchmark_yaml["vectordb"])
    assert params.type == "qdrant"


def test_search_params(benchmark_yaml):
    params = SearchParams(**benchmark_yaml["search"])
    assert params.topk == 10
    assert params.queries_file == Path("benchmark/queries.txt")


def test_config(benchmark_yaml):
    config = Config(**benchmark_yaml)
    assert isinstance(config.embedding_model, EmbeddingModelConfig)
    assert isinstance(config.vectordb, VectorDatabaseConfig)
    assert isinstance(config.search, SearchParams)
    assert isinstance(config.baseline, BaselineResults)
    assert config.compute_recall is False

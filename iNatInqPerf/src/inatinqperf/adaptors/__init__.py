"""Adaptor registry for vector databases."""

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import Metric
from inatinqperf.adaptors.faiss_adaptor import Faiss
from inatinqperf.adaptors.milvus_adaptor import Milvus
from inatinqperf.adaptors.qdrant_adaptor import Qdrant, QdrantCluster
from inatinqperf.adaptors.weaviate_adaptor import Weaviate, WeaviateCluster
from inatinqperf.adaptors.weaviate_async_adaptor import WeaviateAsync

VECTORDBS = {
    "faiss": Faiss,
    "qdrant.hnsw": Qdrant,
    "qdrant.cluster": QdrantCluster,
    "weaviate.hnsw": Weaviate,
    "weaviate.cluster": WeaviateCluster,
    "weaviate_async.hnsw": WeaviateAsync,
    "milvus": Milvus,
}


__all__ = [
    "VECTORDBS",
    "DataPoint",
    "Faiss",
    "Metric",
    "Milvus",
    "Qdrant",
    "QdrantCluster",
    "Query",
    "SearchResult",
    "VectorDatabase",
    "Weaviate",
    "WeaviateAsync",
    "WeaviateCluster",
]

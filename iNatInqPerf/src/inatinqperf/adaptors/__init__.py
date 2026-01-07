"""Adaptor registry for vector databases."""

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import Metric
from inatinqperf.adaptors.qdrant_adaptor import Qdrant
from inatinqperf.adaptors.weaviate_adaptor import Weaviate

VECTORDBS = {
    "qdrant": Qdrant,
    "weaviate": Weaviate,
}


__all__ = [
    "VECTORDBS",
    "DataPoint",
    "Metric",
    "Qdrant",
    "Query",
    "SearchResult",
    "VectorDatabase",
    "Weaviate",
]

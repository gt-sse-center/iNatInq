"""Information Retrieval metrics for benchmark evaluation.

This module provides standard IR metrics for evaluating search quality:
- PrecisionAtK: Fraction of top-K retrieved documents that are relevant
- RecallAtK: Fraction of relevant documents that appear in top-K
- MeanAveragePrecision: Average precision at each relevant hit
- MRR: Reciprocal rank of the first relevant result
- NDCG: Normalized Discounted Cumulative Gain for graded relevance

All metrics auto-register with MetricRegistry and can be retrieved by name.

Example:
    ```python
    from core.benchmark.metrics.ir import PrecisionAtK, RecallAtK

    precision = PrecisionAtK(k=10)
    recall = RecallAtK(k=10)

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc7"}

    p_score = precision.compute(retrieved, relevant)  # 0.4 (2/5)
    r_score = recall.compute(retrieved, relevant)     # 0.67 (2/3)
    ```
"""

import math
from collections.abc import Sequence

from core.benchmark.metrics.base import Metric


class PrecisionAtK(Metric):
    """Precision@K metric: fraction of top-K results that are relevant.

    Precision@K = |{relevant} ∩ {top-K retrieved}| / K

    Attributes:
        k: Number of top results to consider (default: 10).
        name: Metric identifier for registry ("precision@k").
        description: Human-readable description.
    """

    name: str = "precision@k"
    description: str = "Fraction of top-K retrieved documents that are relevant"

    def __init__(self, k: int = 10) -> None:
        """Initialize PrecisionAtK metric.

        Args:
            k: Number of top results to consider (default: 10).

        Raises:
            ValueError: If k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k

    def compute(
        self,
        retrieved: Sequence[str],
        relevant: set[str],
        graded: dict[str, int] | None = None,
    ) -> float:
        """Compute Precision@K.

        Args:
            retrieved: Ordered sequence of retrieved document IDs.
            relevant: Set of relevant document IDs.
            graded: Unused (included for interface compatibility).

        Returns:
            Precision@K score between 0.0 and 1.0.
        """
        top_k = list(retrieved)[: self.k]
        relevant_in_top_k = len(set(top_k) & relevant)
        return relevant_in_top_k / self.k


class RecallAtK(Metric):
    """Recall@K metric: fraction of relevant documents in top-K results.

    Recall@K = |{relevant} ∩ {top-K retrieved}| / |{relevant}|

    Attributes:
        k: Number of top results to consider (default: 10).
        name: Metric identifier for registry ("recall@k").
        description: Human-readable description.
    """

    name: str = "recall@k"
    description: str = "Fraction of relevant documents that appear in top-K results"

    def __init__(self, k: int = 10) -> None:
        """Initialize RecallAtK metric.

        Args:
            k: Number of top results to consider (default: 10).

        Raises:
            ValueError: If k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k

    def compute(
        self,
        retrieved: Sequence[str],
        relevant: set[str],
        graded: dict[str, int] | None = None,
    ) -> float:
        """Compute Recall@K.

        Args:
            retrieved: Ordered sequence of retrieved document IDs.
            relevant: Set of relevant document IDs.
            graded: Unused (included for interface compatibility).

        Returns:
            Recall@K score between 0.0 and 1.0.
            Returns 0.0 if relevant set is empty.
        """
        if len(relevant) == 0:
            return 0.0

        top_k = list(retrieved)[: self.k]
        relevant_in_top_k = len(set(top_k) & relevant)
        return relevant_in_top_k / len(relevant)


class MeanAveragePrecision(Metric):
    """Mean Average Precision (MAP) metric.

    MAP computes the average of precision values at each position where
    a relevant document is found. It rewards systems that return relevant
    documents earlier in the ranking.

    MAP = (1/|relevant|) * Σ(Precision@k * rel(k))

    where rel(k) = 1 if document at rank k is relevant, 0 otherwise.

    Attributes:
        name: Metric identifier for registry ("map").
        description: Human-readable description.
    """

    name: str = "map"
    description: str = "Mean Average Precision - average precision at each relevant hit"

    def compute(
        self,
        retrieved: Sequence[str],
        relevant: set[str],
        graded: dict[str, int] | None = None,
    ) -> float:
        """Compute Mean Average Precision.

        Args:
            retrieved: Ordered sequence of retrieved document IDs.
            relevant: Set of relevant document IDs.
            graded: Unused (included for interface compatibility).

        Returns:
            MAP score between 0.0 and 1.0.
            Returns 0.0 if no relevant documents exist.
        """
        if len(relevant) == 0:
            return 0.0

        retrieved_list = list(retrieved)
        precision_sum = 0.0
        relevant_found = 0

        for i, doc_id in enumerate(retrieved_list):
            if doc_id in relevant:
                relevant_found += 1
                # Precision at this position (1-indexed rank)
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(relevant)


class MRR(Metric):
    """Mean Reciprocal Rank (MRR) metric.

    MRR returns the reciprocal of the rank of the first relevant document.
    It measures how quickly a system returns a relevant result.

    MRR = 1 / rank_of_first_relevant

    Attributes:
        name: Metric identifier for registry ("mrr").
        description: Human-readable description.
    """

    name: str = "mrr"
    description: str = "Mean Reciprocal Rank - reciprocal of first relevant result rank"

    def compute(
        self,
        retrieved: Sequence[str],
        relevant: set[str],
        graded: dict[str, int] | None = None,
    ) -> float:
        """Compute Mean Reciprocal Rank.

        Args:
            retrieved: Ordered sequence of retrieved document IDs.
            relevant: Set of relevant document IDs.
            graded: Unused (included for interface compatibility).

        Returns:
            MRR score between 0.0 and 1.0.
            Returns 0.0 if no relevant documents found in results.
        """
        retrieved_list = list(retrieved)

        for i, doc_id in enumerate(retrieved_list):
            if doc_id in relevant:
                # Rank is 1-indexed
                return 1.0 / (i + 1)

        return 0.0


class NDCG(Metric):
    """Normalized Discounted Cumulative Gain (NDCG) metric.

    NDCG measures ranking quality using graded relevance scores. It accounts
    for the position of relevant documents, giving higher weight to relevant
    documents appearing earlier in the ranking.

    DCG@k = Σ (rel_i / log2(i + 2)) for i = 0 to k-1
    IDCG@k = DCG with ideal ordering (sorted by relevance descending)
    NDCG@k = DCG@k / IDCG@k

    Attributes:
        k: Number of top results to consider (default: 10).
        name: Metric identifier for registry ("ndcg").
        description: Human-readable description.
    """

    name: str = "ndcg"
    description: str = "Normalized Discounted Cumulative Gain - graded relevance metric"

    def __init__(self, k: int = 10) -> None:
        """Initialize NDCG metric.

        Args:
            k: Number of top results to consider (default: 10).
        """
        self.k = k

    def _dcg(self, relevances: list[float]) -> float:
        """Compute Discounted Cumulative Gain for a list of relevance scores.

        Args:
            relevances: List of relevance scores in ranking order.

        Returns:
            DCG score.
        """
        dcg = 0.0
        for i, rel in enumerate(relevances[: self.k]):
            # Discount factor: log2(i + 2) to handle i=0 case
            dcg += rel / math.log2(i + 2)
        return dcg

    def compute(
        self,
        retrieved: Sequence[str],
        relevant: set[str],
        graded: dict[str, int] | None = None,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        Args:
            retrieved: Ordered sequence of retrieved document IDs.
            relevant: Set of relevant document IDs.
            graded: Optional dict mapping document IDs to relevance grades.
                   If None, uses binary relevance (1 for relevant, 0 otherwise).

        Returns:
            NDCG score between 0.0 and 1.0.
            Returns 0.0 if k <= 0 or IDCG is 0 (no relevant documents).
        """
        if self.k <= 0:
            return 0.0

        retrieved_list = list(retrieved)[: self.k]

        # Get relevance scores for retrieved documents
        if graded is not None:
            # Use graded relevance scores
            relevances = [float(graded.get(doc_id, 0)) for doc_id in retrieved_list]
            # For IDCG, use all graded scores (including those not retrieved)
            ideal_relevances = sorted([float(score) for score in graded.values()], reverse=True)
        else:
            # Binary relevance: 1 if in relevant set, 0 otherwise
            relevances = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved_list]
            # For IDCG with binary relevance, ideal is all 1s for each relevant doc
            ideal_relevances = [1.0] * len(relevant)

        # Compute DCG for actual ranking
        dcg = self._dcg(relevances)

        # Compute IDCG (ideal DCG with perfect ranking)
        idcg = self._dcg(ideal_relevances)

        # Avoid division by zero
        if idcg == 0:
            return 0.0

        return dcg / idcg

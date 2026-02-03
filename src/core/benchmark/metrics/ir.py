"""Information Retrieval metrics for benchmark evaluation.

This module provides standard IR metrics for evaluating search quality:
- PrecisionAtK: Fraction of top-K retrieved documents that are relevant
- RecallAtK: Fraction of relevant documents that appear in top-K
- MeanAveragePrecision: Average precision at each relevant hit
- MRR: Reciprocal rank of the first relevant result

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

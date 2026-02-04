"""Information Retrieval metrics for benchmark evaluation.

This module provides standard IR metrics for evaluating search quality:
- PrecisionAtK: Fraction of top-K retrieved documents that are relevant
- RecallAtK: Fraction of relevant documents that appear in top-K

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

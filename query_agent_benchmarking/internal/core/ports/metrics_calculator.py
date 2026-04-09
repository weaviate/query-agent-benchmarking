"""Port interfaces for metrics calculation."""

from typing import Protocol, runtime_checkable

from query_agent_benchmarking.internal.core.models import (
    QueryResult,
    InMemoryQuery,
    AskResult,
)


@runtime_checkable
class SearchMetricsCalculator(Protocol):
    """Protocol for computing search evaluation metrics."""

    def compute(
        self,
        results: list[QueryResult],
        ground_truths: list[InMemoryQuery],
    ) -> dict:
        """Compute search metrics for a set of query results.

        Args:
            results: List of search query results.
            ground_truths: List of queries with ground truth document IDs.

        Returns:
            Dictionary of metric names to scores/aggregates.
        """
        ...


@runtime_checkable
class AskMetricsCalculator(Protocol):
    """Protocol for computing ask evaluation metrics."""

    def compute(self, results: list[AskResult]) -> dict:
        """Compute ask metrics for a set of query results.

        Args:
            results: List of ask query results with system answers.

        Returns:
            Dictionary of metric names to scores/aggregates.
        """
        ...

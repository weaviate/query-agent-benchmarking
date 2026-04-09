"""Port interface for result persistence."""

from typing import Any, Protocol, runtime_checkable

from query_agent_benchmarking.internal.core.domain.models import QueryResult, AskResult


@runtime_checkable
class ResultRepository(Protocol):
    """Protocol for persisting benchmark results."""

    def save_trial_results(
        self,
        results: list[QueryResult],
        config: dict[str, Any],
        trial_number: int,
    ) -> None:
        """Save raw search query results for a single trial."""
        ...

    def save_ask_trial_results(
        self,
        results: list[AskResult],
        config: dict[str, Any],
        trial_number: int,
        alignment_scores: list[int] | None = None,
    ) -> None:
        """Save raw ask query results for a single trial."""
        ...

    def save_trial_metrics(
        self,
        metrics: dict[str, Any],
        config: dict[str, Any],
        trial_number: int,
    ) -> None:
        """Save computed metrics for a single trial."""
        ...

    def save_aggregated_results(
        self,
        aggregated_metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Save aggregated metrics across all trials."""
        ...

"""JSON file-based ResultRepository implementation.

Wraps the existing result_serialization module behind the ResultRepository port.
"""

from typing import Any

from query_agent_benchmarking.internal.core.domain.models import QueryResult, AskResult
from query_agent_benchmarking.internal.adapters.results.serialization import (
    save_trial_results as _save_trial_results,
    save_ask_trial_results as _save_ask_trial_results,
    save_trial_metrics as _save_trial_metrics,
    save_aggregated_results as _save_aggregated_results,
)


class JsonFileResultRepository:
    """Persists benchmark results as JSON files.

    Delegates to the existing result_serialization module.
    """

    def save_trial_results(
        self,
        results: list[QueryResult],
        config: dict[str, Any],
        trial_number: int,
    ) -> None:
        _save_trial_results(results, config, trial_number)

    def save_ask_trial_results(
        self,
        results: list[AskResult],
        config: dict[str, Any],
        trial_number: int,
        alignment_scores: list[int] | None = None,
    ) -> None:
        _save_ask_trial_results(results, config, trial_number, alignment_scores)

    def save_trial_metrics(
        self,
        metrics: dict[str, Any],
        config: dict[str, Any],
        trial_number: int,
    ) -> None:
        _save_trial_metrics(metrics, config, trial_number)

    def save_aggregated_results(
        self,
        aggregated_metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        _save_aggregated_results(aggregated_metrics, config)

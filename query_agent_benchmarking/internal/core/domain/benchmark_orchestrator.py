"""
Benchmark orchestrators for search and ask evaluation.

These orchestrators coordinate the benchmark flow using port interfaces
(dependency injection). They have no direct dependencies on infrastructure.
"""

from typing import Any

from query_agent_benchmarking.internal.core.domain.models import (
    InMemoryQuery,
    InMemoryAskQuery,
)
from query_agent_benchmarking.internal.core.domain.analysis import aggregate_metrics


class SearchBenchmarkOrchestrator:
    """Orchestrates search benchmark execution using injected port implementations.

    Coordinates: load queries -> execute queries -> compute metrics -> save results.
    """

    def __init__(
        self,
        query_runner,
        metrics_calculator,
        result_repository,
    ):
        """
        Args:
            query_runner: Callable that runs search queries (sync or async).
            metrics_calculator: SearchMetricsCalculator port implementation.
            result_repository: ResultRepository port implementation.
        """
        self.query_runner = query_runner
        self.metrics_calculator = metrics_calculator
        self.result_repository = result_repository

    def run_trial(
        self,
        queries: list[InMemoryQuery],
        agent: Any,
        config: dict[str, Any],
        trial_number: int,
    ) -> dict:
        """Run a single search benchmark trial.

        Args:
            queries: Search queries with ground truth.
            agent: SearchAgent port implementation.
            config: Serialization config dictionary.
            trial_number: Current trial number (1-indexed).

        Returns:
            Per-trial metrics dictionary.
        """
        results = self.query_runner(queries, agent)
        self.result_repository.save_trial_results(results, config, trial_number)

        metrics = self.metrics_calculator.compute(results, queries)
        self.result_repository.save_trial_metrics(metrics, config, trial_number)

        return metrics

    def run_and_aggregate(
        self,
        queries: list[InMemoryQuery],
        agent: Any,
        config: dict[str, Any],
        num_trials: int = 1,
    ) -> dict:
        """Run multiple trials and aggregate results.

        Args:
            queries: Search queries with ground truth.
            agent: SearchAgent port implementation.
            config: Serialization config dictionary.
            num_trials: Number of trials to run.

        Returns:
            Aggregated metrics dictionary.
        """
        metrics_across_trials = []
        for trial in range(1, num_trials + 1):
            print(f"\n{'=' * 50}")
            print(f"Trial {trial}/{num_trials}")
            print(f"{'=' * 50}")
            trial_metrics = self.run_trial(queries, agent, config, trial)
            metrics_across_trials.append(trial_metrics)

        aggregated = aggregate_metrics(metrics_across_trials)
        self.result_repository.save_aggregated_results(aggregated, config)
        return aggregated


class AskBenchmarkOrchestrator:
    """Orchestrates ask benchmark execution using injected port implementations.

    Coordinates: load queries -> execute queries -> compute metrics -> save results.
    """

    def __init__(
        self,
        query_runner,
        metrics_calculator,
        result_repository,
    ):
        """
        Args:
            query_runner: Callable that runs ask queries (sync or async).
            metrics_calculator: AskMetricsCalculator port implementation.
            result_repository: ResultRepository port implementation.
        """
        self.query_runner = query_runner
        self.metrics_calculator = metrics_calculator
        self.result_repository = result_repository

    def run_trial(
        self,
        queries: list[InMemoryAskQuery],
        agent: Any,
        config: dict[str, Any],
        trial_number: int,
    ) -> dict:
        """Run a single ask benchmark trial.

        Args:
            queries: Ask queries with ground truth answers.
            agent: AskAgent port implementation.
            config: Serialization config dictionary.
            trial_number: Current trial number (1-indexed).

        Returns:
            Per-trial metrics dictionary.
        """
        results = self.query_runner(queries, agent)

        metrics = self.metrics_calculator.compute(results)

        alignment_scores = metrics.get(
            "alignment_score_scores",
            metrics.get("exact_match_accuracy_scores",
                        metrics.get("officeqa_accuracy_scores")),
        )
        judge_reasonings = metrics.get("judge_reasonings")
        self.result_repository.save_ask_trial_results(
            results, config, trial_number, alignment_scores, judge_reasonings
        )
        self.result_repository.save_trial_metrics(metrics, config, trial_number)

        return metrics

    def run_and_aggregate(
        self,
        queries: list[InMemoryAskQuery],
        agent: Any,
        config: dict[str, Any],
        num_trials: int = 1,
    ) -> dict:
        """Run multiple trials and aggregate results.

        Args:
            queries: Ask queries with ground truth answers.
            agent: AskAgent port implementation.
            config: Serialization config dictionary.
            num_trials: Number of trials to run.

        Returns:
            Aggregated metrics dictionary.
        """
        metrics_across_trials = []
        for trial in range(1, num_trials + 1):
            print(f"\n{'=' * 50}")
            print(f"Trial {trial}/{num_trials}")
            print(f"{'=' * 50}")
            trial_metrics = self.run_trial(queries, agent, config, trial)
            metrics_across_trials.append(trial_metrics)

        aggregated = aggregate_metrics(metrics_across_trials)
        self.result_repository.save_aggregated_results(aggregated, config)
        return aggregated

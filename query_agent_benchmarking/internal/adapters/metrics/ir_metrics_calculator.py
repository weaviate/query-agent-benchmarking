"""SearchMetricsCalculator adapter using IR metric functions.

Implements the SearchMetricsCalculator port by dispatching MetricSpec names
to concrete metric functions from the IR metrics module.
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from query_agent_benchmarking.internal.core.models import QueryResult, InMemoryQuery
from query_agent_benchmarking.internal.core.metrics_config import MetricSpec, resolve_metrics_profile
from query_agent_benchmarking.metrics.ir_metrics import (
    calculate_recall_at_k,
    calculate_success_at_k,
    calculate_nDCG_at_k,
    calculate_coverage,
    calculate_alpha_ndcg,
)

# Maps MetricSpec.name to the concrete metric function
_METRIC_FUNCTIONS = {
    "recall": calculate_recall_at_k,
    "success": calculate_success_at_k,
    "nDCG": calculate_nDCG_at_k,
    "coverage": calculate_coverage,
    "alpha_ndcg": calculate_alpha_ndcg,
}


class IRMetricsCalculator:
    """SearchMetricsCalculator implementation using IR metric functions.

    Resolves the appropriate metrics for a dataset and computes them
    over a set of search results.
    """

    def __init__(self, dataset_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self.metric_specs = resolve_metrics_profile(dataset_name)

    def compute(
        self,
        results: list[QueryResult],
        ground_truths: list[InMemoryQuery],
    ) -> dict:
        """Compute IR metrics for search results.

        Args:
            results: List of search query results.
            ground_truths: List of queries with ground truth document IDs.

        Returns:
            Dictionary with per-metric averages, per-query scores, and timing.
        """
        metric_results: dict[str, list] = {}
        for spec in self.metric_specs:
            metric_results[spec.key] = []

        query_times = []

        for i, (result, ground_truth) in enumerate(
            tqdm(zip(results, ground_truths), desc="Analyzing search results")
        ):
            if result.retrieved_ids == []:
                print(f"\n\033[91mSkipping analysis for query {i} due to error.\033[0m")
                continue

            retrieved_ids = [res.object_id for res in result.retrieved_ids]

            for spec in self.metric_specs:
                func = _METRIC_FUNCTIONS.get(spec.name)
                if func is None:
                    raise ValueError(f"Unknown metric: {spec.name}")

                score = self._compute_single_metric(
                    func, spec, retrieved_ids, ground_truth
                )
                metric_results[spec.key].append(score)

            query_times.append(result.time_taken)

            if (i + 1) % 10 == 0:
                print(f"\n\033[93m--- Analysis Progress ({i + 1}/{len(results)}) ---\033[0m")
                for metric_name, scores in metric_results.items():
                    if scores:
                        display_name = metric_name.replace("_", " ").title()
                        print(f"Current average {display_name}: {np.mean(scores):.2f}")
                print(f"Current average query time: {np.mean(query_times):.2f} seconds")

        results_dict = {
            "avg_query_time": np.mean(query_times) if query_times else 0,
            "query_times": query_times,
        }

        for metric_name, scores in metric_results.items():
            results_dict[f"avg_{metric_name}"] = np.mean(scores) if scores else 0
            results_dict[f"{metric_name}_scores"] = scores

        print("\n\033[92m===== Search Benchmark Results =====\033[0m")
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of queries: {len(results)}")
        for metric_name, scores in metric_results.items():
            if scores:
                display_name = metric_name.replace("_", " ").title()
                print(f"Average {display_name}: {np.mean(scores):.2f}")
        print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")

        return results_dict

    @staticmethod
    def _compute_single_metric(func, spec: MetricSpec, retrieved_ids, ground_truth):
        """Dispatch a single metric computation based on its name."""
        params = spec.params

        if spec.name in ("recall", "success", "nDCG"):
            return func(
                target_ids=ground_truth.dataset_ids,
                retrieved_ids=retrieved_ids,
                **params,
            )
        elif spec.name in ("coverage", "alpha_ndcg"):
            if ground_truth.nugget_data:
                return func(
                    retrieved_ids=retrieved_ids,
                    nugget_data=ground_truth.nugget_data,
                    **params,
                )
            return 0.0
        else:
            try:
                return func(
                    target_ids=ground_truth.dataset_ids,
                    retrieved_ids=retrieved_ids,
                    **params,
                )
            except TypeError:
                return func(retrieved_ids=retrieved_ids, **params)

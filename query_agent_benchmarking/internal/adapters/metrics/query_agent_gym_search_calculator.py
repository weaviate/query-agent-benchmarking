"""SearchMetricsCalculator for QueryAgentGym search mode.

Scores each query two ways:

- **Results**: IR metrics (nDCG@k, from the dataset's metrics profile) of the
  returned record UUIDs against the query's relevant records.
- **Plan**: query recall of the searches the agent issued against the exact
  ground-truth query (see :mod:`query_exact_match`): 1 when the ground-truth
  search was issued, extra searches allowed. Strict exact match, sub-query
  recall/precision, and breakdowns per retrieval shape, environment and
  operator class are reported alongside.

Queries whose agent call failed are left out of every metric, as in ask mode,
and counted under ``num_errors``.
"""

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from query_agent_benchmarking.internal.adapters.metrics.ir_metrics_calculator import IRMetricsCalculator
from query_agent_benchmarking.internal.adapters.metrics.query_exact_match import match_issued_queries
from query_agent_benchmarking.internal.core.domain.models import InMemoryQuery, QueryResult


def _group_means(scores: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {group: float(np.mean(values)) for group, values in sorted(scores.items())}


class QueryAgentGymSearchCalculator:
    """Scores QueryAgentGym search mode results (IR metrics) and plans (query recall).

    Args:
        dataset_name: The dataset name, which selects the IR metrics profile.
        collection_aliases: Maps Weaviate collection names back to dataset
            collection names (e.g. ``{"SurfBeaches_Default": "SurfBeaches"}``).
        array_properties: Each dataset collection's array property names.
        date_ranges: Each dataset collection's ``(min, max)`` day per date property.
        extra_metrics: Extra IR metrics, as in ``IRMetricsCalculator``.
    """

    def __init__(
        self,
        dataset_name: str,
        collection_aliases: Optional[dict[str, str]] = None,
        array_properties: Optional[dict[str, frozenset[str]]] = None,
        date_ranges: Optional[dict[str, dict[str, tuple[str, str]]]] = None,
        extra_metrics: Optional[list[dict]] = None,
    ):
        self.ir_calculator = IRMetricsCalculator(dataset_name=dataset_name, extra_metrics=extra_metrics)
        self.collection_aliases = collection_aliases or {}
        self.array_properties = array_properties or {}
        self.date_ranges = date_ranges or {}

    def compute(self, results: list[QueryResult], ground_truths: list[InMemoryQuery]) -> dict[str, Any]:
        valid = [i for i, r in enumerate(results) if r.error is None]
        num_errors = len(results) - len(valid)
        if num_errors:
            print(f"\033[93m{num_errors} queries failed and are left out of every metric.\033[0m")

        metrics: dict[str, Any] = self.ir_calculator.compute(
            [results[i] for i in valid], [ground_truths[i] for i in valid],
        ) if valid else {"avg_query_time": 0, "query_times": []}
        # Realign per-query IR scores with ``results`` (None for failed queries).
        ir_keys = [spec.key for spec in self.ir_calculator.metric_specs]
        for key in ir_keys:
            aligned: list[Optional[float]] = [None] * len(results)
            for i, score in zip(valid, metrics.get(f"{key}_scores", [])):
                aligned[i] = float(score)
            metrics[f"{key}_scores"] = aligned

        recall_scores: list[Optional[int]] = [None] * len(results)
        exact_scores: list[Optional[int]] = [None] * len(results)
        details: list[Optional[dict]] = [None] * len(results)
        sub_recalls, sub_precisions = [], []
        by_shape: dict[str, list[int]] = {}
        by_environment: dict[str, list[int]] = {}
        by_class: dict[str, list[int]] = {}
        ir_by_shape: dict[str, dict[str, list[float]]] = {key: {} for key in ir_keys}
        unreported = 0

        for i in valid:
            result = results[i]
            query = result.query
            if query.ground_truth_queries is None:
                raise ValueError(
                    f"Query {i} has no ground_truth_queries; QueryAgentGym search "
                    "scoring requires exact ground-truth queries."
                )
            if result.searches is None:
                unreported += 1
            match = match_issued_queries(
                query.ground_truth_queries, result.searches, None,
                self.collection_aliases, self.array_properties, self.date_ranges,
            )
            recall = 1 if match["recall_match"] else 0
            recall_scores[i] = recall
            exact_scores[i] = 1 if match["exact_match"] else 0
            sub_recalls.append(match["recall"])
            sub_precisions.append(match["precision"])

            metadata = query.metadata or {}
            shape = metadata.get("retrieval_shape")
            if shape:
                by_shape.setdefault(shape, []).append(recall)
                for key in ir_keys:
                    ir_by_shape[key].setdefault(shape, []).append(metrics[f"{key}_scores"][i])
            if metadata.get("environment"):
                by_environment.setdefault(metadata["environment"], []).append(recall)
            for operator_class in metadata.get("operator_classes", []):
                by_class.setdefault(operator_class, []).append(recall)

            details[i] = {
                "query_recall": recall,
                "query_exact_match": exact_scores[i],
                "sub_query_recall": match["recall"],
                "sub_query_precision": match["precision"],
                "ground_truth_queries": match["ground_truth_queries"],
                "predicted_queries": match["predicted_queries"],
                "missing_queries": match["missing"],
                "extra_queries": match["extra"],
                **{key: metrics[f"{key}_scores"][i] for key in ir_keys},
            }

        if unreported:
            print(f"\033[93mWarning: {unreported} results carry no searches; "
                  "this agent may not report the searches it issues.\033[0m")

        valid_recall = [s for s in recall_scores if s is not None]
        valid_exact = [s for s in exact_scores if s is not None]
        metrics.update({
            "metric": "query_recall",
            "num_errors": num_errors,
            "avg_query_recall": float(np.mean(valid_recall)) if valid_recall else 0,
            "query_recall_scores": recall_scores,
            "avg_query_exact_match": float(np.mean(valid_exact)) if valid_exact else 0,
            "query_exact_match_scores": exact_scores,
            "avg_sub_query_recall": float(np.mean(sub_recalls)) if sub_recalls else 0,
            "avg_sub_query_precision": float(np.mean(sub_precisions)) if sub_precisions else 0,
            "retrieval_shape_accuracy": _group_means(by_shape),
            "environment_accuracy": _group_means(by_environment),
            "type_accuracy": _group_means(by_class),
            "type_counts": {c: len(v) for c, v in sorted(by_class.items())},
            "query_match_details": details,
        })
        for key in ir_keys:
            metrics[f"retrieval_shape_{key}"] = _group_means(ir_by_shape[key])

        self._print_summary(metrics, ir_keys, len(valid))
        return metrics

    @staticmethod
    def _print_summary(metrics: dict[str, Any], ir_keys: list[str], num_evaluated: int) -> None:
        print("\n\033[92m===== QueryAgentGym Search Mode Results =====\033[0m")
        print(f"Number of queries evaluated: {num_evaluated} ({metrics['num_errors']} failed)")
        print(f"Query Recall: {metrics['avg_query_recall']:.2%}")
        print(f"Query Exact Match (no extra searches): {metrics['avg_query_exact_match']:.2%}")
        for key in ir_keys:
            print(f"{key}: {metrics.get(f'avg_{key}', 0):.4f}")
        shapes = metrics["retrieval_shape_accuracy"]
        if shapes:
            print("\n\033[93mPer Retrieval Shape (query recall | " + " | ".join(ir_keys) + "):\033[0m")
            for shape, acc in shapes.items():
                ir = " | ".join(f"{metrics[f'retrieval_shape_{key}'][shape]:.4f}" for key in ir_keys)
                print(f"  {shape}: {acc:.2%} | {ir}")
        if len(metrics["environment_accuracy"]) > 1:
            print("\n\033[93mPer-Environment Query Recall:\033[0m")
            for environment, acc in metrics["environment_accuracy"].items():
                print(f"  {environment}: {acc:.2%}")

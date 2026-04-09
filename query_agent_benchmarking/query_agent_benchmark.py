"""Backward-compatible facade for benchmark functions.

All implementations now live in domain/ and adapters/ modules.
This module re-exports public functions for backward compatibility.
"""

# Re-export query execution functions from domain layer
from query_agent_benchmarking.domain.query_execution import (  # noqa: F401
    run_search_queries,
    run_search_queries_async,
    run_ask_queries,
    run_ask_queries_async,
)

# Re-export aggregate_metrics from domain layer
from query_agent_benchmarking.domain.analysis import aggregate_metrics  # noqa: F401

# Re-export the original analyze functions for backward compatibility.
# These are async functions that used to live here and are still referenced
# by external consumers. They delegate to the adapter metric calculators.

from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from query_agent_benchmarking.metrics.ir_metrics import (
    calculate_recall_at_k,
    calculate_success_at_k,
    calculate_coverage,
    calculate_alpha_ndcg,
    calculate_nDCG_at_k,
)
from query_agent_benchmarking.metrics.lmjudge_alignment import LMJudge
from query_agent_benchmarking.metrics.exact_match import calculate_exact_match
from query_agent_benchmarking.metrics.officeqa_metric import (
    score_answer as officeqa_score_answer,
    extract_final_answer,
)
from query_agent_benchmarking.models import (
    QueryResult,
    InMemoryQuery,
    InMemoryAskQuery,
    AskResult,
)


# ============================================================================
# Dataset-to-Metrics Registry (kept for backward compatibility)
# ============================================================================

_DEFAULT_METRICS = [
    {"func": calculate_recall_at_k, "params": {"k": 1}},
    {"func": calculate_recall_at_k, "params": {"k": 5}},
    {"func": calculate_recall_at_k, "params": {"k": 20}},
    {"func": calculate_nDCG_at_k, "params": {"k": 10}},
]

DATASET_METRICS: dict[str, list[dict]] = {
    "enron": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
    ],
    "wixqa": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
    ],
    "freshstack-": [
        {"func": calculate_recall_at_k, "params": {"k": 50}},
        {"func": calculate_coverage, "params": {"k": 5}},
        {"func": calculate_coverage, "params": {"k": 10}},
        {"func": calculate_coverage, "params": {"k": 20}},
        {"func": calculate_alpha_ndcg, "params": {"alpha": 0.5, "k": 10}},
    ],
    "beir/": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "lotte/": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_success_at_k, "params": {"k": 5}},
    ],
    "bright/": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "irpapers": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
    ],
    "vidore_v3_hr": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "longmemeval-s": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 10}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "longmemeval-m": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 10}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
    "reasonir-biology-subset": [
        {"func": calculate_recall_at_k, "params": {"k": 1}},
        {"func": calculate_recall_at_k, "params": {"k": 5}},
        {"func": calculate_recall_at_k, "params": {"k": 20}},
        {"func": calculate_nDCG_at_k, "params": {"k": 10}},
    ],
}


def _resolve_metrics(dataset_name: Optional[str]) -> list[dict]:
    """Resolve the metrics config for a given dataset name using prefix matching."""
    if dataset_name is None:
        return _DEFAULT_METRICS
    for prefix, metrics in DATASET_METRICS.items():
        if dataset_name == prefix or dataset_name.startswith(prefix):
            return metrics
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# Analysis functions (kept for backward compatibility)
# ============================================================================

async def analyze_search_results(
    results: list[QueryResult],
    ground_truths: list[InMemoryQuery],
    dataset_name: Optional[str] = None,
):
    """Analyze search results with dataset-specific IR metrics.

    .. deprecated::
        Use ``adapters.metrics.ir_metrics_calculator.IRMetricsCalculator`` instead.
    """
    metrics = _resolve_metrics(dataset_name)

    metric_results = {}
    for config in metrics:
        name = config["func"].__name__.replace("calculate_", "")
        if "k" in config["params"]:
            key = f"{name}_at_{config['params']['k']}"
        else:
            key = name
        metric_results[key] = []

    query_times = []

    for i, (result, ground_truth) in enumerate(tqdm(zip(results, ground_truths), desc="Analyzing search results")):
        if result.retrieved_ids == []:
            print(f"\n\033[91mSkipping analysis for query {i} due to error.\033[0m")
            continue

        retrieved_ids = [res.object_id for res in result.retrieved_ids]

        for metric_config in metrics:
            metric_func = metric_config["func"]
            params = metric_config["params"]
            func_name = metric_func.__name__

            key = func_name.replace("calculate_", "")
            if "k" in params:
                key = f"{key}_at_{params['k']}"

            score = 0.0
            if "recall" in func_name:
                score = metric_func(target_ids=ground_truth.dataset_ids, retrieved_ids=retrieved_ids, **params)
            elif func_name in ["calculate_coverage", "calculate_alpha_ndcg"]:
                if ground_truth.nugget_data:
                    score = metric_func(retrieved_ids=retrieved_ids, nugget_data=ground_truth.nugget_data, **params)
                else:
                    score = 0.0
            elif "nDCG" in func_name or "ndcg" in func_name.lower():
                score = metric_func(target_ids=ground_truth.dataset_ids, retrieved_ids=retrieved_ids, **params)
            else:
                try:
                    score = metric_func(target_ids=ground_truth.dataset_ids, retrieved_ids=retrieved_ids, **params)
                except TypeError:
                    score = metric_func(retrieved_ids=retrieved_ids, **params)

            metric_results[key].append(score)

        query_times.append(result.time_taken)

        if (i + 1) % 10 == 0:
            print(f"\n\033[93m--- Analysis Progress ({i + 1}/{len(results)}) ---\033[0m")
            for metric_name, scores in metric_results.items():
                if scores:
                    display_name = metric_name.replace("_", " ").title()
                    print(f"Current average {display_name}: {np.mean(scores):.2f}")
            print(f"Current average query time: {np.mean(query_times):.2f} seconds")

    results_dict: dict[str, Any] = {
        "avg_query_time": np.mean(query_times) if query_times else 0,
        "query_times": query_times,
    }

    for metric_name, scores in metric_results.items():
        results_dict[f"avg_{metric_name}"] = np.mean(scores) if scores else 0
        results_dict[f"{metric_name}_scores"] = scores

    print("\n\033[92m===== Search Benchmark Results =====\033[0m")
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(results)}")
    for metric_name, scores in metric_results.items():
        if scores:
            display_name = metric_name.replace("_", " ").title()
            print(f"Average {display_name}: {np.mean(scores):.2f}")
    print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")

    return results_dict


async def analyze_ask_results(
    results: list[AskResult],
    judge_model: str = "openai/gpt-4.1",
    ensemble_k: int = 3,
    use_exact_match: bool = False,
    use_officeqa_metric: bool = False,
    officeqa_tolerance: float = 0.00,
) -> dict:
    """Analyze ask results using LLM-as-judge or exact match.

    .. deprecated::
        Use ``adapters.metrics.ask_metrics_calculator`` classes instead.
    """
    if use_officeqa_metric:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with OfficeQA fuzzy match (tolerance={officeqa_tolerance})...\033[0m")
    elif use_exact_match:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with exact match...\033[0m")
    else:
        print(f"\n\033[94mAnalyzing {len(results)} ask results with LLM judge...\033[0m")
        print(f"Judge model: {judge_model}, Ensemble K: {ensemble_k}")

    judge = None if (use_exact_match or use_officeqa_metric) else LMJudge(model=judge_model, ensemble_k=ensemble_k)

    alignment_scores = []
    query_times = []
    misaligned_indices = []
    total_input_tokens = 0
    total_output_tokens = 0

    if use_officeqa_metric:
        desc = "Running OfficeQA fuzzy match"
    elif use_exact_match:
        desc = "Running exact match"
    else:
        desc = "Running LLM judge"

    for i, result in enumerate(tqdm(results, desc=desc)):
        if result.system_answer.startswith("[ERROR]"):
            print(f"\n\033[91mSkipping evaluation for query {i} due to error.\033[0m")
            continue

        if use_officeqa_metric:
            predicted = extract_final_answer(result.system_answer)
            score = officeqa_score_answer(
                ground_truth=result.query.ground_truth_answer,
                predicted=predicted,
                tolerance=officeqa_tolerance,
            )
            aligned = score == 1.0
            alignment_scores.append(1 if aligned else 0)
        elif use_exact_match:
            aligned = calculate_exact_match(
                system_answer=result.system_answer,
                ground_truth_answer=result.query.ground_truth_answer,
            )
            alignment_scores.append(1 if aligned else 0)
        else:
            judge_result = judge.evaluate_with_details(
                question=result.query.question,
                system_answer=result.system_answer,
                correct_answer=result.query.ground_truth_answer,
            )
            aligned = judge_result["aligned"]
            alignment_scores.append(1 if aligned else 0)
            total_input_tokens += judge_result.get("input_tokens", 0)
            total_output_tokens += judge_result.get("output_tokens", 0)

        query_times.append(result.time_taken)
        if not aligned:
            misaligned_indices.append(i)

        if (i + 1) % 5 == 0:
            current_avg = np.mean(alignment_scores) if alignment_scores else 0
            metric_name = "exact match" if use_exact_match else "alignment"
            print(f"\n\033[93m--- Evaluation Progress ({i + 1}/{len(results)}) ---\033[0m")
            print(f"Running {metric_name} score: {current_avg:.2%}")
            print(f"Running avg query time: {np.mean(query_times):.2f}s")

    if use_officeqa_metric:
        metric_name = "officeqa_accuracy"
        metric_label = "officeqa_fuzzy_match"
    elif use_exact_match:
        metric_name = "exact_match_accuracy"
        metric_label = "exact_match"
    else:
        metric_name = "alignment_score"
        metric_label = "llm_judge"
    metric_results = {metric_name: alignment_scores}

    results_dict: dict[str, Any] = {
        "avg_query_time": np.mean(query_times) if query_times else 0,
        "query_times": query_times,
        "misaligned_indices": misaligned_indices,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "metric": metric_label,
    }

    for name, scores in metric_results.items():
        results_dict[f"avg_{name}"] = np.mean(scores) if scores else 0
        results_dict[f"{name}_scores"] = scores

    if not use_exact_match:
        results_dict["judge_model"] = judge_model
        results_dict["ensemble_k"] = ensemble_k

    print("\n\033[92m===== Ask Benchmark Results =====\033[0m")
    print(f"Number of queries evaluated: {len(alignment_scores)}")
    for name, scores in metric_results.items():
        if scores:
            display_name = name.replace("_", " ").title()
            print(f"Average {display_name}: {np.mean(scores):.2%}")
    print(f"Misaligned queries: {len(misaligned_indices)}")
    print(f"Average Query Time: {results_dict['avg_query_time']:.2f} seconds")
    if not use_exact_match:
        print(f"Total Judge Tokens: {total_input_tokens + total_output_tokens}")

    return results_dict

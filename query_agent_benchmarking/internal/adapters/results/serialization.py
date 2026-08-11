import os
from pathlib import Path
from datetime import datetime
from typing import Any
import json
from query_agent_benchmarking.internal.core.domain.models import QueryResult, AskResult
from query_agent_benchmarking.internal.core.domain.metrics_config import (
    resolve_primary_metric,
    ASK_METRIC_KEY_MAP,
)


# All results are saved to console/results/ at the project root
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "console" / "results"


def _ensure_results_dir() -> None:
    """Create the results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_run_id(config: dict[str, Any]) -> str:
    """Get or create a stable run ID for this benchmark run.

    The run ID is generated once and cached in the config dict so that all
    files produced by the same run share the same identifier.
    """
    if "run_id" not in config:
        config["run_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config["run_id"]


def _build_base_path(config: dict[str, Any]) -> str:
    """Build the base filename (without extension) from config."""
    dataset_identifier = config["dataset_identifier"]
    agent_name = config["agent_name"]
    num_trials = config.get("num_trials", 1)
    output_path = config.get("output_path")

    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        run_id = _get_run_id(config)
        return f"{dataset_name_for_file}-{agent_name}-{num_trials}-{run_id}-results"
    else:
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
        return os.path.splitext(os.path.basename(output_path))[0]


def save_trial_results(
    results: list[QueryResult],
    config: dict[str, Any],
    trial_number: int,
) -> None:
    """
    Save raw query results for a single search trial.

    Args:
        results: List of query results from the trial
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
        trial_number: Current trial number (1-indexed)
    """
    _ensure_results_dir()
    base_path = _build_base_path(config)
    trial_output_path = RESULTS_DIR / f"{base_path}-trial-{trial_number}.json"

    metadata = {
        "dataset": config["dataset_identifier"],
        "agent_name": config["agent_name"],
        "trial_number": trial_number,
        "total_queries": len(results),
        "timestamp": datetime.now().isoformat(),
        "mode": "search",
    }
    for key in ("effort", "filtering", "sweep_id"):
        if config.get(key) is not None:
            metadata[key] = config[key]

    trial_data = {
        "metadata": metadata,
        "queries": [
            _search_query_to_dict(idx, result)
            for idx, result in enumerate(results)
        ]
    }

    with open(trial_output_path, "w") as f:
        json.dump(trial_data, f, indent=2)


def _search_query_to_dict(idx: int, result: QueryResult) -> dict[str, Any]:
    """Serialize a single search QueryResult to a JSON-ready dict.

    The agent's search plan (``searches``/``num_searches``) is only included
    when the agent reported one. Agents that don't expose a plan (direct
    hybrid/vector/BM25 or external services that haven't implemented it) leave
    ``result.searches`` as ``None``, and these keys are omitted entirely.
    """
    query_dict: dict[str, Any] = {
        "query_id": f"q{idx}",
        "question": result.query.question,
        "ground_truth_ids": result.query_ground_truth_id,
        "retrieved_ids": [obj.object_id for obj in result.retrieved_ids],
        "num_retrieved": len(result.retrieved_ids),
        "num_ground_truth": len(result.query_ground_truth_id),
        "time_taken": result.time_taken,
    }
    if result.searches is not None:
        # The agent's search plan: how it decomposed this query into structured
        # sub-searches (query text, filters, sort, uuid).
        query_dict["num_searches"] = len(result.searches)
        query_dict["searches"] = [s.model_dump(mode="json") for s in result.searches]
    return query_dict


def save_ask_trial_results(
    results: list[AskResult],
    config: dict[str, Any],
    trial_number: int,
    alignment_scores: list[int] | None = None,
    judge_reasonings: list[str | list[dict] | None] | None = None,
) -> None:
    """
    Save raw query results for a single ask trial.

    Args:
        results: List of AskResult objects from the trial
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
        trial_number: Current trial number (1-indexed)
        alignment_scores: Optional list of per-query scores (1=correct, 0=incorrect)
        judge_reasonings: Optional list of per-query judge reasoning strings
    """
    _ensure_results_dir()
    base_path = _build_base_path(config)
    trial_output_path = RESULTS_DIR / f"{base_path}-trial-{trial_number}.json"

    queries = []
    failed_query_ids = []
    misaligned_query_ids = []

    for idx, result in enumerate(results):
        query_id = f"q{idx}"
        is_error = result.system_answer.startswith("[ERROR]")

        query_data = {
            "query_id": query_id,
            "question": result.query.question,
            "ground_truth_answer": result.query.ground_truth_answer,
            "system_answer": result.system_answer,
            "time_taken": result.time_taken,
            "is_error": is_error,
        }
        if result.query.oracle_context_id:
            query_data["oracle_context_id"] = result.query.oracle_context_id
        if result.query.tenant_id:
            query_data["tenant_id"] = result.query.tenant_id
        if result.query.question_type:
            query_data["question_type"] = result.query.question_type
        if alignment_scores and idx < len(alignment_scores) and alignment_scores[idx] is not None:
            query_data["score"] = alignment_scores[idx]
            if not is_error and alignment_scores[idx] == 0:
                misaligned_query_ids.append(query_id)
        if judge_reasonings and idx < len(judge_reasonings) and judge_reasonings[idx]:
            query_data["judge_reasoning"] = judge_reasonings[idx]
        if result.retrieved_context is not None:
            try:
                json.dumps(result.retrieved_context)  # verify serializable
                query_data["retrieved_context"] = result.retrieved_context
            except (TypeError, ValueError):
                pass  # skip non-serializable context
        if is_error:
            failed_query_ids.append(query_id)

        queries.append(query_data)

    trial_data = {
        "metadata": {
            "dataset": config["dataset_identifier"],
            "agent_name": config["agent_name"],
            "trial_number": trial_number,
            "total_queries": len(results),
            "total_errors": len(failed_query_ids),
            "total_misaligned": len(misaligned_query_ids),
            "timestamp": datetime.now().isoformat(),
            "mode": "ask",
        },
        "failed_query_ids": failed_query_ids,
        "misaligned_query_ids": misaligned_query_ids,
        "queries": queries,
    }

    with open(trial_output_path, "w") as f:
        json.dump(trial_data, f, indent=2)


def save_trial_metrics(
    metrics: dict[str, Any],
    config: dict[str, Any],
    trial_number: int,
) -> None:
    """
    Save metrics for a single trial.

    Args:
        metrics: Dictionary of computed metrics
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
        trial_number: Current trial number (1-indexed)
    """
    _ensure_results_dir()
    base_path = _build_base_path(config)
    metrics_output_path = RESULTS_DIR / f"{base_path}-trial-{trial_number}-metrics.json"

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def _resolve_key_metric(aggregated_metrics: dict[str, Any], dataset_identifier: str) -> str | None:
    """Determine the headline metric for an experiment.

    For search experiments we look up the dataset's primary metric from the
    metrics config registry.  For ask experiments we detect which calculator
    was used from the metric keys present in the aggregated results.

    Returns the key as it appears in the aggregated dict (e.g.
    ``avg_nDCG_at_10_mean``), or ``None`` if it cannot be determined.
    """
    # Ask-mode detection: check for known ask metric keys
    for ask_key in ASK_METRIC_KEY_MAP.values():
        candidate = f"avg_{ask_key}_mean"
        if candidate in aggregated_metrics:
            return candidate

    # Search-mode: use the dataset's primary metric from the registry
    try:
        primary = resolve_primary_metric(dataset_identifier)
    except (ValueError, TypeError):
        primary = None

    if primary:
        candidate = f"avg_{primary}_mean"
        if candidate in aggregated_metrics:
            return candidate

    return None


def save_aggregated_results(
    aggregated_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """
    Save aggregated metrics across all trials.

    Args:
        aggregated_metrics: Dictionary of aggregated metrics
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
    """
    _ensure_results_dir()

    dataset_identifier = config["dataset_identifier"]
    agent_name = config["agent_name"]
    num_trials = config.get("num_trials", 1)
    output_path = config.get("output_path")
    use_async = config.get("use_async", True)
    batch_size = config.get("batch_size")
    max_concurrent = config.get("max_concurrent")

    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        run_id = _get_run_id(config)
        filename = f"{dataset_name_for_file}-{agent_name}-{num_trials}-{run_id}-results.json"
    else:
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
        filename = os.path.basename(output_path)

    full_output_path = RESULTS_DIR / filename

    # Determine the key (headline) metric for this experiment.
    key_metric = _resolve_key_metric(aggregated_metrics, dataset_identifier)

    # Add metadata
    aggregated_metrics["timestamp"] = datetime.now().isoformat()
    if key_metric:
        aggregated_metrics["key_metric"] = key_metric
    aggregated_metrics["config"] = {
        "dataset": dataset_identifier,
        "agent_name": agent_name,
        "num_trials": num_trials,
        "use_async": use_async,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
    }
    # Query Agent search-mode settings, when set. `sweep_id` ties the
    # low/medium/high runs of one effort sweep together so the console can
    # group them.
    for key in ("effort", "filtering", "sweep_id"):
        if config.get(key) is not None:
            aggregated_metrics["config"][key] = config[key]

    with open(full_output_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)

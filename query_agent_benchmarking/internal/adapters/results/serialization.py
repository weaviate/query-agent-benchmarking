import os
from pathlib import Path
from datetime import datetime
from typing import Any
import json
from query_agent_benchmarking.internal.core.models import QueryResult, AskResult


# All results are saved to the result-visualizer/results/ directory
RESULTS_DIR = Path(__file__).parent.parent / "result-visualizer" / "results"


def _ensure_results_dir() -> None:
    """Create the results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _build_base_path(config: dict[str, Any]) -> str:
    """Build the base filename (without extension) from config."""
    dataset_identifier = config["dataset_identifier"]
    agent_name = config["agent_name"]
    num_trials = config.get("num_trials", 1)
    output_path = config.get("output_path")

    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        return f"{dataset_name_for_file}-{agent_name}-{num_trials}-results"
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

    trial_data = {
        "metadata": {
            "dataset": config["dataset_identifier"],
            "agent_name": config["agent_name"],
            "trial_number": trial_number,
            "total_queries": len(results),
            "timestamp": datetime.now().isoformat(),
            "mode": "search",
        },
        "queries": [
            {
                "query_id": f"q{idx}",
                "question": result.query.question,
                "ground_truth_ids": result.query_ground_truth_id,
                "retrieved_ids": [obj.object_id for obj in result.retrieved_ids],
                "num_retrieved": len(result.retrieved_ids),
                "num_ground_truth": len(result.query_ground_truth_id),
                "time_taken": result.time_taken,
            }
            for idx, result in enumerate(results)
        ]
    }

    with open(trial_output_path, "w") as f:
        json.dump(trial_data, f, indent=2)


def save_ask_trial_results(
    results: list[AskResult],
    config: dict[str, Any],
    trial_number: int,
    alignment_scores: list[int] | None = None,
) -> None:
    """
    Save raw query results for a single ask trial.

    Args:
        results: List of AskResult objects from the trial
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
        trial_number: Current trial number (1-indexed)
        alignment_scores: Optional list of per-query scores (1=correct, 0=incorrect)
    """
    _ensure_results_dir()
    base_path = _build_base_path(config)
    trial_output_path = RESULTS_DIR / f"{base_path}-trial-{trial_number}.json"

    queries = []
    for idx, result in enumerate(results):
        query_data = {
            "query_id": f"q{idx}",
            "question": result.query.question,
            "ground_truth_answer": result.query.ground_truth_answer,
            "system_answer": result.system_answer,
            "time_taken": result.time_taken,
        }
        if result.query.oracle_context_id:
            query_data["oracle_context_id"] = result.query.oracle_context_id
        if alignment_scores and idx < len(alignment_scores):
            query_data["score"] = alignment_scores[idx]
        queries.append(query_data)

    trial_data = {
        "metadata": {
            "dataset": config["dataset_identifier"],
            "agent_name": config["agent_name"],
            "trial_number": trial_number,
            "total_queries": len(results),
            "timestamp": datetime.now().isoformat(),
            "mode": "ask",
        },
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
        filename = f"{dataset_name_for_file}-{agent_name}-{num_trials}-results.json"
    else:
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
        filename = os.path.basename(output_path)

    full_output_path = RESULTS_DIR / filename

    # Add metadata
    aggregated_metrics["timestamp"] = datetime.now().isoformat()
    aggregated_metrics["config"] = {
        "dataset": dataset_identifier,
        "agent_name": agent_name,
        "num_trials": num_trials,
        "use_async": use_async,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
    }

    with open(full_output_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)

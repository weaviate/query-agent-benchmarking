import os
from datetime import datetime
from typing import Dict, Any, List
import json
from query_agent_benchmarking.models import QueryResult


def save_trial_results(
    results: List[QueryResult],
    config: Dict[str, Any],
    trial_number: int,
) -> None:
    """
    Save raw query results for a single trial.
    
    Args:
        results: List of query results from the trial
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
        trial_number: Current trial number (1-indexed)
    
    Returns:
        Path to the saved file
    """
    # Extract from config
    dataset_identifier = config["dataset_identifier"]
    agent_name = config["agent_name"]
    num_trials = config.get("num_trials", 1)
    output_path = config.get("output_path")
    
    # Generate base path
    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        base_path = f"{dataset_name_for_file}-{agent_name}-{num_trials}-results"
    else:
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
        base_path = os.path.splitext(output_path)[0]
    
    # Create trial-specific path
    trial_output_path = f"{base_path}-trial-{trial_number}.json"
    
    trial_data = {
        "metadata": {
            "dataset": dataset_identifier,
            "agent_name": agent_name,
            "trial_number": trial_number,
            "total_queries": len(results),
            "timestamp": datetime.now().isoformat(),
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


def save_trial_metrics(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    trial_number: int,
) -> None:
    """
    Save metrics for a single trial.
    
    Args:
        metrics: Dictionary of computed metrics
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
        trial_number: Current trial number (1-indexed)
    
    Returns:
        Path to the saved file
    """
    # Extract from config
    dataset_identifier = config["dataset_identifier"]
    agent_name = config["agent_name"]
    num_trials = config.get("num_trials", 1)
    output_path = config.get("output_path")
    
    # Generate base path
    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        base_path = f"{dataset_name_for_file}-{agent_name}-{num_trials}-results"
    else:
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
        base_path = os.path.splitext(output_path)[0]
    
    # Create metrics-specific path
    metrics_output_path = f"{base_path}-trial-{trial_number}-metrics.json"
    
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
def save_aggregated_results(
    aggregated_metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Save aggregated metrics across all trials.
    
    Args:
        aggregated_metrics: Dictionary of aggregated metrics
        config: Configuration dictionary (must contain dataset_identifier, agent_name, etc.)
    
    Returns:
        Path to the saved file
    """
    # Extract from config
    dataset_identifier = config["dataset_identifier"]
    agent_name = config["agent_name"]
    num_trials = config.get("num_trials", 1)
    output_path = config.get("output_path")
    use_async = config.get("use_async", True)
    batch_size = config.get("batch_size")
    max_concurrent = config.get("max_concurrent")
    
    # Generate output path
    if output_path is None:
        dataset_name_for_file = dataset_identifier.replace("/", "-")
        output_path = f"{dataset_name_for_file}-{agent_name}-{num_trials}-results.json"
    else:
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
    
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
    
    with open(output_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    
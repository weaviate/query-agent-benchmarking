"""
Domain analysis functions for aggregating benchmark results.

Contains aggregate_metrics (pure math, no external dependencies).
The per-trial analysis functions (analyze_search_results, analyze_ask_results)
are in query_agent_benchmark.py and will be refactored to use ports in Phase 2.
"""

import numpy as np


def _aggregate_type_accuracy(metrics_across_trials: list[dict]) -> dict | None:
    """Aggregate per-type accuracy dicts across trials.

    Returns a dict mapping each question type to {mean, std, min, max, raw}
    or None if no trials contain type_accuracy.
    """
    trials_with_types = [
        t["type_accuracy"] for t in metrics_across_trials
        if isinstance(t.get("type_accuracy"), dict) and t["type_accuracy"]
    ]
    if not trials_with_types:
        return None

    all_types = sorted({k for d in trials_with_types for k in d})
    aggregated = {}
    for qtype in all_types:
        values = [d[qtype] for d in trials_with_types if qtype in d]
        if values:
            aggregated[qtype] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "raw": values,
            }
    return aggregated


def aggregate_metrics(metrics_across_trials: list[dict]) -> dict:
    """Aggregate metrics from multiple trials into statistical summaries.

    Args:
        metrics_across_trials: List of per-trial metrics dictionaries.

    Returns:
        Dictionary with mean, std, min, max for each avg_ metric,
        per-type accuracy aggregation (if present), and per-trial summaries.
    """
    if not metrics_across_trials:
        return {}

    avg_keys = [k for k in metrics_across_trials[0].keys() if k.startswith("avg_")]

    aggregated = {
        "num_trials": len(metrics_across_trials),
        "trials": [],
    }

    for key in avg_keys:
        values = [trial[key] for trial in metrics_across_trials if key in trial]
        if values:
            metric_name = key
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
            aggregated[f"{metric_name}_min"] = float(np.min(values))
            aggregated[f"{metric_name}_max"] = float(np.max(values))

    type_agg = _aggregate_type_accuracy(metrics_across_trials)
    if type_agg:
        aggregated["type_accuracy"] = type_agg

    for i, trial in enumerate(metrics_across_trials):
        trial_summary = {
            "trial": i + 1,
            **{k: v for k, v in trial.items() if k.startswith("avg_")},
        }
        if isinstance(trial.get("type_accuracy"), dict):
            trial_summary["type_accuracy"] = trial["type_accuracy"]
        aggregated["trials"].append(trial_summary)

    # --- Print ---

    print("\n" + "=" * 70)
    print(f"AGGREGATED RESULTS ({len(metrics_across_trials)} trials)")
    print("=" * 70)

    for key in avg_keys:
        values = [trial[key] for trial in metrics_across_trials if key in trial]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            metric_display = key.replace("_", " ").title()

            print(f"\n{metric_display}:")
            print(f"  Mean: {mean:.4f} (+/- {std:.4f})")
            print(f"  Min:  {min_val:.4f}")
            print(f"  Max:  {max_val:.4f}")
            print(f"  Raw:  {[f'{v:.4f}' for v in values]}")

    if type_agg:
        print(f"\nPer-Type Accuracy ({len(metrics_across_trials)} trials):")
        for qtype, stats in type_agg.items():
            print(f"  {qtype}:")
            print(f"    Mean: {stats['mean']:.2%} (+/- {stats['std']:.2%})")
            print(f"    Raw:  {[f'{v:.2%}' for v in stats['raw']]}")

    print("\n" + "=" * 70 + "\n")

    return aggregated

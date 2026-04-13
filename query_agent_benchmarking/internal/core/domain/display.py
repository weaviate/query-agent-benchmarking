"""Display and formatting utilities."""

import inspect
from typing import Any

from query_agent_benchmarking.internal.core.domain.models import InMemoryQuery


def get_object_by_dataset_id(dataset_id, objects_list):
    """Retrieve an object by its dataset_id from the objects list."""
    for obj in objects_list:
        if obj["dataset_id"] == dataset_id:
            return obj
    return None


def make_json_serializable(obj):
    """Convert objects to JSON serializable formats."""
    if hasattr(obj, '__dict__'):
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()
                if not k.startswith('_') and not inspect.ismethod(v)}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except Exception:
            return f"<Non-serializable object of type {type(obj).__name__}>"


def pretty_print_in_memory_query(in_memory_query: InMemoryQuery):
    """Pretty print an InMemoryQuery with colored output."""
    print(f"\t\033[96mQuestion\033[0m: {in_memory_query.question}")
    print(f"\t\033[96mDataset IDs\033[0m: {in_memory_query.dataset_ids}")
    print("=" * 60)
    print("\n\n")


def pretty_print_in_memory_document(in_memory_document_object: dict):
    """Pretty print an in-memory document with colored output."""
    print(f"Dataset ID: {in_memory_document_object['dataset_id']}")
    if "content" in in_memory_document_object:
        print(f"\t\033[96mDocument\033[0m: {in_memory_document_object['content']}")
    print("=" * 60)
    print("\n\n")


def print_results_comparison(all_results: dict[str, dict[str, Any]]) -> None:
    """Print key metrics for each agent."""
    if not all_results:
        return

    all_mean_metrics = set()
    for results in all_results.values():
        if "error" not in results:
            for key, value in results.items():
                if key.endswith("_mean") and isinstance(value, (int, float)):
                    all_mean_metrics.add(key)

    if not all_mean_metrics:
        return

    priority_order = ["recall", "ndcg", "precision", "mrr", "query_time"]

    def metric_priority(key):
        for i, term in enumerate(priority_order):
            if term in key.lower():
                return (i, key)
        return (len(priority_order), key)

    sorted_metrics = sorted(all_mean_metrics, key=metric_priority)

    def format_name(key):
        name = key.replace("avg_", "").replace("_mean", "")
        name = name.replace("recall_at_", "Recall@").replace("ndcg_at_k", "NDCG@10")
        name = name.replace("query_time", "Time(s)")
        return name

    print("\nResults:")
    for agent_name, results in all_results.items():
        print(f"\n{agent_name}:")

        if "error" in results:
            print("  ERROR")
            continue

        for metric_key in sorted_metrics:
            value = results.get(metric_key)
            if value is not None:
                print(f"  {format_name(metric_key)}: {value:.3f}")

    print()


def print_suite_results(suite_results: dict[str, dict[str, Any]]) -> None:
    """Print a comparison table of metrics across multiple datasets."""
    if not suite_results:
        return

    # Collect all mean metrics across datasets
    all_mean_metrics: set[str] = set()
    for results in suite_results.values():
        if "error" not in results:
            for key, value in results.items():
                if key.endswith("_mean") and isinstance(value, (int, float)):
                    all_mean_metrics.add(key)

    if not all_mean_metrics:
        print("\nNo metrics to compare across datasets.")
        return

    priority_order = ["recall", "ndcg", "precision", "mrr", "query_time"]

    def metric_priority(key):
        for i, term in enumerate(priority_order):
            if term in key.lower():
                return (i, key)
        return (len(priority_order), key)

    sorted_metrics = sorted(all_mean_metrics, key=metric_priority)

    def format_name(key):
        name = key.replace("avg_", "").replace("_mean", "")
        name = name.replace("recall_at_", "Recall@").replace("ndcg_at_k", "NDCG@10")
        name = name.replace("query_time", "Time(s)")
        return name

    print("\n" + "=" * 60)
    print("Suite Results (across datasets)")
    print("=" * 60)

    for dataset_name, results in suite_results.items():
        print(f"\n\033[92m{dataset_name}\033[0m:")

        if "error" in results:
            print(f"  \033[91mERROR: {results['error']}\033[0m")
            continue

        for metric_key in sorted_metrics:
            value = results.get(metric_key)
            if value is not None:
                print(f"  {format_name(metric_key)}: {value:.3f}")

    print()

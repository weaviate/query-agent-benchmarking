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
    # Raw docs name their ID differently per dataset; the DatasetSpec maps
    # them to `dataset_id` only at insert time.
    for id_field in ("dataset_id", "doc_id", "id", "corpus_id", "docid"):
        if id_field in in_memory_document_object:
            print(f"Dataset ID: {in_memory_document_object[id_field]}")
            break
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


# ANSI colors, matching the style used elsewhere in this module.
_GREEN = "\033[92m"
_RED = "\033[91m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Canonical ordering for effort levels (unknown levels are appended as-is).
_EFFORT_ORDER = ["medium", "high", "ultrahigh"]


def _ordered_levels(results_by_effort: dict[str, Any]) -> list[str]:
    """Order effort levels medium -> ultrahigh, with any unknown levels appended."""
    levels = [e for e in _EFFORT_ORDER if e in results_by_effort]
    levels += [e for e in results_by_effort if e not in levels]
    return levels


def _effort_metric_label(base: str) -> str:
    """``avg_recall@1`` -> ``recall@1`` (drop the avg_ prefix for display)."""
    return base[len("avg_"):] if base.startswith("avg_") else base


def _effort_means_and_stds(entry: dict[str, Any]) -> tuple[dict, dict]:
    """Split an aggregated-metrics dict into {base: mean} and {base: std}.

    Aggregated keys look like ``avg_recall@1_mean`` / ``avg_recall@1_std``;
    both returned dicts are keyed by the shared ``avg_recall@1`` base.
    """
    metrics = entry.get("metrics", {}) or {}
    means, stds = {}, {}
    for key, value in metrics.items():
        if key.endswith("_mean") and isinstance(value, (int, float)):
            base = key[: -len("_mean")]
            means[base] = value
            std = metrics.get(f"{base}_std")
            if isinstance(std, (int, float)):
                stds[base] = std
    return means, stds


def print_effort_comparison(
    results_by_effort: dict[str, dict[str, Any]],
    dataset: str | None = None,
) -> None:
    """Print a metric x effort comparison table for a single dataset.

    ``results_by_effort`` maps an effort level ("medium"|"high"|"ultrahigh") to
    ``{"metrics": <aggregated metrics>, "seconds": float, "error": str | None}``.
    The best (highest) value per metric is highlighted, a ``Δ(high−low)`` column
    summarizes the effect of effort, and wall-clock per level is printed below.
    """
    levels = _ordered_levels(results_by_effort)
    if not levels:
        return

    means = {e: _effort_means_and_stds(results_by_effort[e])[0] for e in levels}
    stds = {e: _effort_means_and_stds(results_by_effort[e])[1] for e in levels}

    # Metric rows in first-seen order across levels.
    metric_order: list[str] = []
    for e in levels:
        for base in means[e]:
            if base not in metric_order:
                metric_order.append(base)

    title = "EFFORT COMPARISON"
    if dataset:
        title += f" — {dataset}"
    print("\n" + "=" * 72)
    print(f"{_BOLD}{title}{_RESET}")
    print("=" * 72)

    if not metric_order:
        print(f"{_RED}No metrics to compare (all effort levels failed?).{_RESET}")
        for e in levels:
            err = results_by_effort[e].get("error")
            if err:
                print(f"  {e}: {err}")
        print()
        return

    name_w = max([len("Metric")] + [len(_effort_metric_label(b)) for b in metric_order])
    col_w = 16
    has_delta = "medium" in levels and "ultrahigh" in levels
    delta_label = "Δ(ultrahigh−medium)"
    delta_w = max(col_w, len(delta_label))

    header = "Metric".ljust(name_w) + "".join(e.center(col_w) for e in levels)
    if has_delta:
        header += delta_label.rjust(delta_w)
    print("\n" + header)
    print("-" * len(header))

    for base in metric_order:
        values = {e: means[e].get(base) for e in levels}
        present = [v for v in values.values() if v is not None]
        best = max(present) if present else None

        row = _effort_metric_label(base).ljust(name_w)
        for e in levels:
            value = values[e]
            if value is None:
                row += "—".center(col_w)
                continue
            std = stds[e].get(base)
            cell = f"{value:.4f}" + (f" ±{std:.3f}" if std is not None else "")
            if best is not None and value == best:
                # pad to account for the invisible ANSI codes
                row += f"{_GREEN}{cell}{_RESET}".center(col_w + len(_GREEN) + len(_RESET))
            else:
                row += cell.center(col_w)

        if has_delta:
            low, high = values["medium"], values["ultrahigh"]
            row += (f"{high - low:+.4f}" if low is not None and high is not None else "—").rjust(delta_w)
        print(row)

    print(f"\n{_DIM}Wall-clock:{_RESET}")
    for e in levels:
        seconds = results_by_effort[e].get("seconds")
        note = "  (failed)" if results_by_effort[e].get("error") else ""
        secs = f"{seconds:8.1f}s" if isinstance(seconds, (int, float)) else "       —"
        print(f"  {e.ljust(name_w)} {secs}{note}")
    print("=" * 72 + "\n")


def print_effort_suite(suite_results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Print effort comparisons for every dataset in a multi-dataset sweep.

    ``suite_results`` maps a dataset name to its ``results_by_effort`` dict.
    Each dataset gets its own comparison table.
    """
    if not suite_results:
        return
    print("\n" + "=" * 72)
    print(f"{_BOLD}EFFORT SWEEP — {len(suite_results)} dataset(s){_RESET}")
    print("=" * 72)
    for dataset, results_by_effort in suite_results.items():
        # A dataset that failed before its sweep ran maps to {"error": "..."}
        # (see run_search_evals) rather than a per-effort results dict.
        error = results_by_effort.get("error")
        if isinstance(error, str):
            print(f"\n{_RED}{dataset} failed: {error}{_RESET}")
            continue
        print_effort_comparison(results_by_effort, dataset=dataset)

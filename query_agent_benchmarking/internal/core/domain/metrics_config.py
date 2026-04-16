"""
Metrics configuration for the domain layer.

Defines which metrics apply to which datasets as pure data (no function references).
The adapter layer maps MetricSpec names to actual metric implementations.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric computation."""
    name: str
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Generate a human-readable key for this metric (e.g., 'recall_at_5')."""
        if "k" in self.params:
            return f"{self.name}_at_{self.params['k']}"
        return self.name


@dataclass(frozen=True)
class MetricsProfile:
    """A set of metrics to compute for a specific dataset pattern."""
    dataset_pattern: str
    metrics: tuple[MetricSpec, ...]
    primary_metric: str | None = None


# ============================================================================
# Default Metrics Registry (data only, no function references)
# ============================================================================

_DEFAULT_METRICS = (
    MetricSpec("recall", {"k": 1}),
    MetricSpec("recall", {"k": 5}),
    MetricSpec("recall", {"k": 20}),
    MetricSpec("nDCG", {"k": 10}),
)

DATASET_METRICS_REGISTRY: tuple[MetricsProfile, ...] = (
    MetricsProfile("enron", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
    ), primary_metric="recall_at_5"),
    MetricsProfile("wixqa", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
    ), primary_metric="recall_at_5"),
    MetricsProfile("freshstack-", (
        MetricSpec("recall", {"k": 50}),
        MetricSpec("coverage", {"k": 5}),
        MetricSpec("coverage", {"k": 10}),
        MetricSpec("coverage", {"k": 20}),
        MetricSpec("alpha_ndcg", {"alpha": 0.5, "k": 10}),
    ), primary_metric="alpha_ndcg_at_10"),
    MetricsProfile("beir/", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    ), primary_metric="nDCG_at_10"),
    MetricsProfile("lotte/", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("success", {"k": 5}),
    ), primary_metric="success_at_5"),
    MetricsProfile("bright/", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    ), primary_metric="nDCG_at_10"),
    MetricsProfile("irpapers", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
    ), primary_metric="recall_at_5"),
    MetricsProfile("vidore_v3_hr", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    ), primary_metric="nDCG_at_10"),
    MetricsProfile("longmemeval-s", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 10}),
        MetricSpec("nDCG", {"k": 10}),
    ), primary_metric="nDCG_at_10"),
    MetricsProfile("longmemeval-m", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 10}),
        MetricSpec("nDCG", {"k": 10}),
    ), primary_metric="nDCG_at_10"),
    MetricsProfile("reasonir-biology-subset", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    ), primary_metric="nDCG_at_10"),
)


_DEFAULT_PRIMARY_METRIC = "nDCG_at_10"

# Ask-mode calculators write their own metric key. Map the calculator's
# "metric" field to the aggregated-result key that serves as the headline.
ASK_METRIC_KEY_MAP: dict[str, str] = {
    "llm_judge": "alignment_score",
    "exact_match": "exact_match_accuracy",
    "officeqa_fuzzy_match": "officeqa_accuracy",
    "longmemeval_judge": "alignment_score",
}


def resolve_primary_metric(dataset_name: str | None) -> str:
    """Return the primary (headline) metric key for a dataset.

    The returned key matches the suffix used in the aggregated results file,
    e.g. ``"nDCG_at_10"`` which appears as ``avg_nDCG_at_10_mean``.
    """
    if dataset_name is None:
        return _DEFAULT_PRIMARY_METRIC

    for profile in DATASET_METRICS_REGISTRY:
        if dataset_name == profile.dataset_pattern or dataset_name.startswith(profile.dataset_pattern):
            return profile.primary_metric or _DEFAULT_PRIMARY_METRIC

    return _DEFAULT_PRIMARY_METRIC


def resolve_metrics_profile(
    dataset_name: str | None,
    extra_metrics: tuple[MetricSpec, ...] | None = None,
) -> tuple[MetricSpec, ...]:
    """Resolve the metrics specs for a given dataset name using prefix matching.

    Args:
        dataset_name: The dataset name to look up, or None for defaults.
        extra_metrics: Additional MetricSpec instances to append to the
            dataset's default set. Duplicates (same key) are skipped.

    Returns:
        Tuple of MetricSpec instances for the dataset.

    Raises:
        ValueError: If no matching profile is found for a non-None dataset name.
    """
    if dataset_name is None:
        base = _DEFAULT_METRICS
    else:
        base = None
        for profile in DATASET_METRICS_REGISTRY:
            if dataset_name == profile.dataset_pattern or dataset_name.startswith(profile.dataset_pattern):
                base = profile.metrics
                break
        if base is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    if not extra_metrics:
        return base

    existing_keys = {spec.key for spec in base}
    merged = list(base)
    for spec in extra_metrics:
        if spec.key not in existing_keys:
            merged.append(spec)
            existing_keys.add(spec.key)
    return tuple(merged)


def parse_extra_metrics(raw: list[dict] | None) -> tuple[MetricSpec, ...] | None:
    """Parse extra_metrics from a config dict (YAML or kwargs).

    Args:
        raw: List of dicts with "name" and "params" keys, e.g.
            ``[{"name": "recall", "params": {"k": 50}}]``.

    Returns:
        Tuple of MetricSpec instances, or None if input is empty/None.
    """
    if not raw:
        return None
    return tuple(MetricSpec(entry["name"], entry.get("params", {})) for entry in raw)

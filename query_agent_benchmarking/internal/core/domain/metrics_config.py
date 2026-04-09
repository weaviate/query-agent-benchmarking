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
    )),
    MetricsProfile("wixqa", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
    )),
    MetricsProfile("freshstack-", (
        MetricSpec("recall", {"k": 50}),
        MetricSpec("coverage", {"k": 5}),
        MetricSpec("coverage", {"k": 10}),
        MetricSpec("coverage", {"k": 20}),
        MetricSpec("alpha_ndcg", {"alpha": 0.5, "k": 10}),
    )),
    MetricsProfile("beir/", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    )),
    MetricsProfile("lotte/", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("success", {"k": 5}),
    )),
    MetricsProfile("bright/", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    )),
    MetricsProfile("irpapers", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
    )),
    MetricsProfile("vidore_v3_hr", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    )),
    MetricsProfile("longmemeval-s", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 10}),
        MetricSpec("nDCG", {"k": 10}),
    )),
    MetricsProfile("longmemeval-m", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 10}),
        MetricSpec("nDCG", {"k": 10}),
    )),
    MetricsProfile("reasonir-biology-subset", (
        MetricSpec("recall", {"k": 1}),
        MetricSpec("recall", {"k": 5}),
        MetricSpec("recall", {"k": 20}),
        MetricSpec("nDCG", {"k": 10}),
    )),
)


def resolve_metrics_profile(dataset_name: str | None) -> tuple[MetricSpec, ...]:
    """Resolve the metrics specs for a given dataset name using prefix matching.

    Args:
        dataset_name: The dataset name to look up, or None for defaults.

    Returns:
        Tuple of MetricSpec instances for the dataset.

    Raises:
        ValueError: If no matching profile is found for a non-None dataset name.
    """
    if dataset_name is None:
        return _DEFAULT_METRICS

    for profile in DATASET_METRICS_REGISTRY:
        if dataset_name == profile.dataset_pattern or dataset_name.startswith(profile.dataset_pattern):
            return profile.metrics

    raise ValueError(f"Unknown dataset: {dataset_name}")

"""Tests for MetricSpec, MetricsProfile, and the DATASET_METRICS_REGISTRY."""

import pytest

from query_agent_benchmarking.internal.core.domain.metrics_config import (
    MetricSpec,
    resolve_metrics_profile,
    parse_extra_metrics,
    DATASET_METRICS_REGISTRY,
)


class TestMetricSpec:
    def test_key_with_k(self):
        spec = MetricSpec("recall", {"k": 5})
        assert spec.key == "recall_at_5"

    def test_key_without_k(self):
        spec = MetricSpec("alpha_ndcg", {"alpha": 0.5})
        assert spec.key == "alpha_ndcg"

    def test_frozen(self):
        spec = MetricSpec("recall", {"k": 1})
        with pytest.raises(AttributeError):
            spec.name = "something"


class TestResolveMetricsProfile:
    def test_none_returns_defaults(self):
        metrics = resolve_metrics_profile(None)
        assert len(metrics) == 4
        names = [m.name for m in metrics]
        assert "recall" in names
        assert "nDCG" in names

    def test_exact_match(self):
        metrics = resolve_metrics_profile("enron")
        assert len(metrics) == 3
        assert all(m.name == "recall" for m in metrics)

    def test_prefix_match_beir(self):
        metrics = resolve_metrics_profile("beir/scifact/test")
        assert len(metrics) == 4

    def test_prefix_match_freshstack(self):
        metrics = resolve_metrics_profile("freshstack-angular")
        assert len(metrics) == 5
        names = [m.name for m in metrics]
        assert "coverage" in names
        assert "alpha_ndcg" in names

    def test_lotte_has_success(self):
        metrics = resolve_metrics_profile("lotte/lifestyle/test/forum")
        names = [m.name for m in metrics]
        assert "success" in names

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            resolve_metrics_profile("nonexistent_dataset")


class TestExtraMetrics:
    def test_extra_metrics_appended(self):
        extra = (MetricSpec("recall", {"k": 50}), MetricSpec("recall", {"k": 100}))
        metrics = resolve_metrics_profile("bright/biology", extra_metrics=extra)
        keys = [m.key for m in metrics]
        assert "recall_at_50" in keys
        assert "recall_at_100" in keys

    def test_extra_metrics_deduplicates(self):
        extra = (MetricSpec("recall", {"k": 1}),)  # already in bright/ profile
        metrics = resolve_metrics_profile("bright/biology", extra_metrics=extra)
        keys = [m.key for m in metrics]
        assert keys.count("recall_at_1") == 1

    def test_extra_metrics_none_is_noop(self):
        base = resolve_metrics_profile("bright/biology")
        with_none = resolve_metrics_profile("bright/biology", extra_metrics=None)
        assert base == with_none

    def test_extra_metrics_empty_is_noop(self):
        base = resolve_metrics_profile("bright/biology")
        with_empty = resolve_metrics_profile("bright/biology", extra_metrics=())
        assert base == with_empty


class TestParseExtraMetrics:
    def test_parse_valid(self):
        raw = [{"name": "recall", "params": {"k": 50}}]
        result = parse_extra_metrics(raw)
        assert len(result) == 1
        assert result[0].key == "recall_at_50"

    def test_parse_none(self):
        assert parse_extra_metrics(None) is None

    def test_parse_empty(self):
        assert parse_extra_metrics([]) is None

    def test_parse_missing_params(self):
        raw = [{"name": "nDCG"}]
        result = parse_extra_metrics(raw)
        assert result[0].params == {}


class TestRegistryCompleteness:
    def test_all_profiles_have_metrics(self):
        for profile in DATASET_METRICS_REGISTRY:
            assert len(profile.metrics) > 0, f"Empty metrics for {profile.dataset_pattern}"

    def test_all_metric_names_are_known(self):
        known_names = {"recall", "precision", "success", "nDCG", "coverage", "alpha_ndcg"}
        for profile in DATASET_METRICS_REGISTRY:
            for spec in profile.metrics:
                assert spec.name in known_names, f"Unknown metric {spec.name} in {profile.dataset_pattern}"

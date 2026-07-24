"""Tests for the effort-sweep display helpers."""

from query_agent_benchmarking.internal.core.domain.display import (
    print_effort_comparison,
    print_effort_suite,
)


def _entry(metrics, seconds=1.0, error=None):
    return {"metrics": metrics, "seconds": seconds, "error": error}


def test_effort_suite_skips_failed_dataset(capsys):
    """A dataset that failed before its sweep ran maps to {"error": str};
    it must be reported without crashing the other datasets' tables."""
    suite = {
        "bright/biology": {"error": "connection refused"},
        "beir/scifact": {
            "low": _entry({"avg_recall_at_5_mean": 0.5, "avg_recall_at_5_std": 0.1}),
            "high": _entry({"avg_recall_at_5_mean": 0.6, "avg_recall_at_5_std": 0.1}),
        },
    }

    print_effort_suite(suite)

    out = capsys.readouterr().out
    assert "bright/biology failed: connection refused" in out
    assert "beir/scifact" in out
    assert "recall_at_5" in out


def test_effort_comparison_shows_zero_std(capsys):
    print_effort_comparison(
        {"low": _entry({"avg_recall_at_5_mean": 0.5, "avg_recall_at_5_std": 0.0})}
    )

    out = capsys.readouterr().out
    assert "±0.000" in out


def test_effort_comparison_all_failed(capsys):
    print_effort_comparison(
        {"low": _entry({}, error="boom"), "high": _entry({}, error="bang")}
    )

    out = capsys.readouterr().out
    assert "No metrics to compare" in out
    assert "boom" in out and "bang" in out

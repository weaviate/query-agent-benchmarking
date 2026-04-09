"""Tests for domain analysis functions."""

import pytest

from query_agent_benchmarking.internal.core.analysis import aggregate_metrics


class TestAggregateMetrics:
    def test_empty_input(self):
        assert aggregate_metrics([]) == {}

    def test_single_trial(self):
        trial = {
            "avg_recall_at_5": 0.8,
            "avg_ndcg_at_k": 0.75,
            "recall_at_5_scores": [0.6, 1.0],
        }
        result = aggregate_metrics([trial])

        assert result["num_trials"] == 1
        assert result["avg_recall_at_5_mean"] == pytest.approx(0.8)
        assert result["avg_recall_at_5_std"] == 0.0
        assert result["avg_ndcg_at_k_mean"] == 0.75
        assert len(result["trials"]) == 1

    def test_multiple_trials(self):
        trials = [
            {"avg_recall_at_5": 0.7, "avg_query_time": 0.5},
            {"avg_recall_at_5": 0.9, "avg_query_time": 0.3},
            {"avg_recall_at_5": 0.8, "avg_query_time": 0.4},
        ]
        result = aggregate_metrics(trials)

        assert result["num_trials"] == 3
        assert result["avg_recall_at_5_mean"] == pytest.approx(0.8)
        assert result["avg_recall_at_5_min"] == 0.7
        assert result["avg_recall_at_5_max"] == 0.9
        assert result["avg_recall_at_5_std"] > 0
        assert len(result["trials"]) == 3

    def test_trial_summaries(self):
        trials = [
            {"avg_recall_at_5": 0.7, "avg_query_time": 0.5},
            {"avg_recall_at_5": 0.9, "avg_query_time": 0.3},
        ]
        result = aggregate_metrics(trials)

        assert result["trials"][0]["trial"] == 1
        assert result["trials"][0]["avg_recall_at_5"] == 0.7
        assert result["trials"][1]["trial"] == 2
        assert result["trials"][1]["avg_recall_at_5"] == 0.9

    def test_non_avg_keys_excluded(self):
        trial = {
            "avg_recall_at_5": 0.8,
            "recall_at_5_scores": [0.6, 1.0],
            "query_times": [0.1, 0.2],
        }
        result = aggregate_metrics([trial])

        # Only avg_ keys get mean/std/min/max
        assert "avg_recall_at_5_mean" in result
        assert "recall_at_5_scores_mean" not in result
        assert "query_times_mean" not in result

"""Tests for the benchmark orchestrator with all-mocked dependencies."""

import pytest

from query_agent_benchmarking.internal.core.benchmark_orchestrator import (
    SearchBenchmarkOrchestrator,
    AskBenchmarkOrchestrator,
)
from query_agent_benchmarking.internal.core.models import (
    InMemoryQuery,
    InMemoryAskQuery,
    ObjectID,
    QueryResult,
    AskResult,
)
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse


# ============================================================================
# Mock Dependencies
# ============================================================================

class MockQueryRunner:
    """Mock that returns pre-configured results."""
    def __init__(self, results):
        self.results = results
        self.call_count = 0

    def __call__(self, queries, agent):
        self.call_count += 1
        return self.results


class MockMetricsCalculator:
    """Mock metrics calculator returning fixed metrics."""
    def __init__(self, metrics=None):
        self.metrics = metrics or {"avg_recall_at_5": 0.8, "avg_query_time": 0.5}
        self.call_count = 0

    def compute(self, results, ground_truths=None):
        self.call_count += 1
        return self.metrics


class MockResultRepo:
    def __init__(self):
        self.trials = []
        self.ask_trials = []
        self.metrics_saved = []
        self.aggregated = []

    def save_trial_results(self, results, config, trial_number):
        self.trials.append(trial_number)

    def save_ask_trial_results(self, results, config, trial_number, alignment_scores=None):
        self.ask_trials.append(trial_number)

    def save_trial_metrics(self, metrics, config, trial_number):
        self.metrics_saved.append((trial_number, metrics))

    def save_aggregated_results(self, aggregated, config):
        self.aggregated.append(aggregated)


# ============================================================================
# Search Orchestrator Tests
# ============================================================================

class TestSearchBenchmarkOrchestrator:
    def _make_fixtures(self):
        queries = [
            InMemoryQuery(question="Q1", dataset_ids=["d1"]),
        ]
        results = [
            QueryResult(
                query=queries[0],
                query_ground_truth_id=["d1"],
                retrieved_ids=[ObjectID(object_id="d1")],
                time_taken=0.1,
            )
        ]
        return queries, results

    def test_single_trial(self):
        queries, results = self._make_fixtures()
        runner = MockQueryRunner(results)
        calc = MockMetricsCalculator()
        repo = MockResultRepo()

        orchestrator = SearchBenchmarkOrchestrator(runner, calc, repo)
        metrics = orchestrator.run_trial(queries, None, {"dataset_identifier": "test", "agent_name": "mock"}, 1)

        assert runner.call_count == 1
        assert calc.call_count == 1
        assert len(repo.trials) == 1
        assert len(repo.metrics_saved) == 1
        assert metrics["avg_recall_at_5"] == 0.8

    def test_multi_trial_aggregation(self):
        queries, results = self._make_fixtures()
        runner = MockQueryRunner(results)
        calc = MockMetricsCalculator()
        repo = MockResultRepo()

        orchestrator = SearchBenchmarkOrchestrator(runner, calc, repo)
        config = {"dataset_identifier": "test", "agent_name": "mock"}
        aggregated = orchestrator.run_and_aggregate(queries, None, config, num_trials=3)

        assert runner.call_count == 3
        assert calc.call_count == 3
        assert len(repo.trials) == 3
        assert len(repo.aggregated) == 1
        assert aggregated["num_trials"] == 3


# ============================================================================
# Ask Orchestrator Tests
# ============================================================================

class TestAskBenchmarkOrchestrator:
    def _make_fixtures(self):
        queries = [
            InMemoryAskQuery(question="Q1", ground_truth_answer="A1"),
        ]
        results = [
            AskResult(query=queries[0], system_answer="A1", time_taken=0.1),
        ]
        return queries, results

    def test_single_trial(self):
        queries, results = self._make_fixtures()
        runner = MockQueryRunner(results)
        calc = MockMetricsCalculator({"avg_alignment_score": 1.0, "avg_query_time": 0.1})
        repo = MockResultRepo()

        orchestrator = AskBenchmarkOrchestrator(runner, calc, repo)
        config = {"dataset_identifier": "test", "agent_name": "mock"}
        metrics = orchestrator.run_trial(queries, None, config, 1)

        assert runner.call_count == 1
        assert calc.call_count == 1
        assert metrics["avg_alignment_score"] == 1.0

    def test_multi_trial_aggregation(self):
        queries, results = self._make_fixtures()
        runner = MockQueryRunner(results)
        calc = MockMetricsCalculator({"avg_alignment_score": 0.9, "avg_query_time": 0.2})
        repo = MockResultRepo()

        orchestrator = AskBenchmarkOrchestrator(runner, calc, repo)
        config = {"dataset_identifier": "test", "agent_name": "mock"}
        aggregated = orchestrator.run_and_aggregate(queries, None, config, num_trials=2)

        assert runner.call_count == 2
        assert len(repo.aggregated) == 1
        assert aggregated["num_trials"] == 2

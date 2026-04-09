"""End-to-end integration tests for search benchmark flow.

These tests verify the full search benchmark pipeline using mock agents,
ensuring that domain logic, metric calculators, and result repositories
work together correctly without requiring a live Weaviate instance.
"""

import pytest
from unittest.mock import MagicMock

from query_agent_benchmarking.internal.core.domain.models import (
    InMemoryQuery,
    ObjectID,
    QueryResult,
)
from query_agent_benchmarking.internal.core.domain.query_execution import run_search_queries
from query_agent_benchmarking.internal.core.domain.analysis import aggregate_metrics
from query_agent_benchmarking.internal.core.domain.benchmark_orchestrator import SearchBenchmarkOrchestrator
from query_agent_benchmarking.internal.adapters.metrics.ir_metrics_calculator import IRMetricsCalculator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_queries():
    return [
        InMemoryQuery(
            question="What is information retrieval?",
            dataset_ids=["doc1", "doc2"],
            query_id="q1",
        ),
        InMemoryQuery(
            question="How does BM25 work?",
            dataset_ids=["doc3"],
            query_id="q2",
        ),
        InMemoryQuery(
            question="What is TF-IDF?",
            dataset_ids=["doc1", "doc4"],
            query_id="q3",
        ),
    ]


class MockSearchAgent:
    """Mock search agent that returns deterministic results."""

    def __init__(self, result_map: dict[str, list[str]]):
        self.result_map = result_map

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        ids = self.result_map.get(query, ["unknown1", "unknown2"])
        return [ObjectID(object_id=doc_id) for doc_id in ids]


# ============================================================================
# Integration tests
# ============================================================================

class TestSearchPipelineE2E:
    """Test the full search pipeline: queries -> execution -> metrics -> aggregation."""

    def test_full_pipeline_with_perfect_retrieval(self, sample_queries):
        """Agent returns all ground truth docs -> perfect recall."""
        agent = MockSearchAgent({
            "What is information retrieval?": ["doc1", "doc2", "extra1"],
            "How does BM25 work?": ["doc3", "extra2"],
            "What is TF-IDF?": ["doc1", "doc4", "extra3"],
        })

        # Step 1: Execute queries
        results = run_search_queries(sample_queries, agent)
        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)

        # Step 2: Compute metrics (BEIR-style)
        calculator = IRMetricsCalculator(dataset_name="beir/scifact/test")
        metrics = calculator.compute(results, sample_queries)

        # All ground truth docs are in top-20, so recall@20 should be 1.0
        assert metrics["avg_recall_at_20"] == pytest.approx(1.0)
        assert "avg_query_time" in metrics
        assert metrics["avg_query_time"] > 0

    def test_full_pipeline_with_partial_retrieval(self, sample_queries):
        """Agent misses some ground truth docs -> partial recall."""
        agent = MockSearchAgent({
            "What is information retrieval?": ["doc1"],  # misses doc2
            "How does BM25 work?": ["wrong_doc"],  # misses doc3
            "What is TF-IDF?": ["doc1", "doc4"],  # perfect
        })

        results = run_search_queries(sample_queries, agent)
        calculator = IRMetricsCalculator(dataset_name="beir/scifact/test")
        metrics = calculator.compute(results, sample_queries)

        # Not perfect recall
        assert metrics["avg_recall_at_20"] < 1.0
        assert metrics["avg_recall_at_20"] > 0.0

    def test_full_pipeline_aggregation(self, sample_queries):
        """Test multi-trial aggregation produces correct structure."""
        agent = MockSearchAgent({
            "What is information retrieval?": ["doc1", "doc2"],
            "How does BM25 work?": ["doc3"],
            "What is TF-IDF?": ["doc1", "doc4"],
        })

        calculator = IRMetricsCalculator(dataset_name="beir/scifact/test")

        # Run 2 trials
        all_metrics = []
        for _ in range(2):
            results = run_search_queries(sample_queries, agent)
            metrics = calculator.compute(results, sample_queries)
            all_metrics.append(metrics)

        aggregated = aggregate_metrics(all_metrics)

        assert aggregated["num_trials"] == 2
        assert "avg_recall_at_20_mean" in aggregated
        assert "avg_recall_at_20_std" in aggregated
        assert len(aggregated["trials"]) == 2

    def test_orchestrator_end_to_end(self, sample_queries):
        """Test SearchBenchmarkOrchestrator with all-mocked dependencies."""
        agent = MockSearchAgent({
            "What is information retrieval?": ["doc1", "doc2"],
            "How does BM25 work?": ["doc3"],
            "What is TF-IDF?": ["doc1", "doc4"],
        })

        calculator = IRMetricsCalculator(dataset_name="beir/scifact/test")
        mock_repo = MagicMock()

        orchestrator = SearchBenchmarkOrchestrator(
            query_runner=run_search_queries,
            metrics_calculator=calculator,
            result_repository=mock_repo,
        )

        config = {"dataset_identifier": "beir/scifact/test", "agent_name": "mock"}
        aggregated = orchestrator.run_and_aggregate(
            queries=sample_queries,
            agent=agent,
            config=config,
            num_trials=2,
        )

        assert aggregated["num_trials"] == 2
        assert mock_repo.save_trial_results.call_count == 2
        assert mock_repo.save_trial_metrics.call_count == 2
        assert mock_repo.save_aggregated_results.call_count == 1

    def test_different_dataset_metrics(self, sample_queries):
        """Different datasets use different metric profiles."""
        agent = MockSearchAgent({
            "What is information retrieval?": ["doc1", "doc2"],
            "How does BM25 work?": ["doc3"],
            "What is TF-IDF?": ["doc1", "doc4"],
        })

        results = run_search_queries(sample_queries, agent)

        # BEIR metrics
        beir_calc = IRMetricsCalculator(dataset_name="beir/scifact/test")
        beir_metrics = beir_calc.compute(results, sample_queries)
        assert "avg_nDCG_at_10" in beir_metrics

        # Enron metrics (no nDCG)
        enron_calc = IRMetricsCalculator(dataset_name="enron")
        enron_metrics = enron_calc.compute(results, sample_queries)
        assert "avg_nDCG_at_10" not in enron_metrics
        assert "avg_recall_at_1" in enron_metrics

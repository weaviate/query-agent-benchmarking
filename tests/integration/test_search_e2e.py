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


class TestZeroAndUnderRetrievalScoring:
    """Regression tests: zero-retrieval queries are scored (not skipped) and
    precision@k uses a constant-k denominator (filtered-cars precision mode).
    """

    @staticmethod
    def _query(ground_truth_ids):
        return InMemoryQuery(question="q", dataset_ids=ground_truth_ids)

    @staticmethod
    def _result(query, retrieved_ids):
        return QueryResult(
            query=query,
            query_ground_truth_id=query.dataset_ids,
            retrieved_ids=[ObjectID(object_id=i) for i in retrieved_ids],
            time_taken=0.1,
        )

    def test_zero_retrieval_is_scored_and_counted(self):
        """A zero-retrieval query scores 0 at every metric and stays in the
        aggregate denominator -- it pulls the averages below 1.0."""
        # Two queries: one perfect, one retrieving nothing.
        queries = [self._query(["doc1"]), self._query(["doc2"])]
        results = [
            self._result(queries[0], ["doc1"]),       # perfect
            self._result(queries[1], []),             # zero retrieval
        ]

        calculator = IRMetricsCalculator(dataset_name="filtered-cars")
        metrics = calculator.compute(results, queries)

        # Per-query scores: 2 entries (denominator = query count = 2),
        # the zero-retrieval query contributes 0.
        for key in ("precision_at_1", "precision_at_5", "precision_at_20",
                    "recall_at_1", "recall_at_5", "recall_at_20"):
            scores = metrics[f"{key}_scores"]
            assert len(scores) == 2, key
            assert scores[1] == 0.0, key

        # Aggregates are the simple mean over both queries -> 0.5, never 1.0.
        assert metrics["avg_precision_at_1"] == pytest.approx(0.5)
        assert metrics["avg_recall_at_1"] == pytest.approx(0.5)

    def test_precision_uses_constant_k_denominator(self):
        """A query retrieving 6 relevant docs scores precision@20 = 6/20 = 0.30,
        not 1.0; recall@5 cannot reach 1.0 when 6 ground-truth docs exist."""
        ground_truth = [f"doc{i}" for i in range(6)]
        query = self._query(ground_truth)
        result = self._result(query, ground_truth)  # 6 retrieved, all relevant

        calculator = IRMetricsCalculator(dataset_name="filtered-cars")
        metrics = calculator.compute([result], [query])

        assert metrics["precision_at_20_scores"][0] == pytest.approx(0.30)
        assert metrics["precision_at_5_scores"][0] == pytest.approx(1.0)  # 5/5
        # Top 5 cannot hold all 6 relevant docs.
        assert metrics["recall_at_5_scores"][0] < 1.0

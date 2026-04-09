"""End-to-end integration tests for ask benchmark flow.

These tests verify the full ask benchmark pipeline using mock agents,
ensuring that domain logic, metric calculators, and result repositories
work together correctly without requiring a live Weaviate or LLM instance.
"""

import pytest
from unittest.mock import MagicMock

from query_agent_benchmarking.internal.core.domain.models import InMemoryAskQuery, AskResult
from query_agent_benchmarking.internal.core.domain.query_execution import run_ask_queries
from query_agent_benchmarking.internal.core.domain.analysis import aggregate_metrics
from query_agent_benchmarking.internal.core.domain.benchmark_orchestrator import AskBenchmarkOrchestrator
from query_agent_benchmarking.internal.adapters.metrics.ask_metrics_calculator import (
    ExactMatchAskCalculator,
)
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ask_queries():
    return [
        InMemoryAskQuery(
            question="What is the capital of France?",
            ground_truth_answer="Paris",
        ),
        InMemoryAskQuery(
            question="What is 2+2?",
            ground_truth_answer="4",
        ),
        InMemoryAskQuery(
            question="Who wrote Hamlet?",
            ground_truth_answer="William Shakespeare",
        ),
    ]


class MockAskAgent:
    """Mock ask agent that returns deterministic answers."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def run(self, query, oracle_context_id=None, tenant_id=None):
        answer = self.answers.get(query, "I don't know")
        return AskResponse(final_answer=answer)


# ============================================================================
# Integration tests
# ============================================================================

class TestAskPipelineE2E:
    """Test the full ask pipeline: queries -> execution -> metrics -> aggregation."""

    def test_full_pipeline_perfect_answers(self, sample_ask_queries):
        """Agent answers all questions correctly -> 100% exact match."""
        agent = MockAskAgent({
            "What is the capital of France?": "Paris",
            "What is 2+2?": "4",
            "Who wrote Hamlet?": "William Shakespeare",
        })

        # Step 1: Execute queries
        results = run_ask_queries(sample_ask_queries, agent)
        assert len(results) == 3
        assert all(isinstance(r, AskResult) for r in results)
        assert results[0].system_answer == "Paris"

        # Step 2: Compute metrics (exact match)
        calculator = ExactMatchAskCalculator()
        metrics = calculator.compute(results)

        assert metrics["avg_exact_match_accuracy"] == pytest.approx(1.0)
        assert len(metrics["exact_match_accuracy_scores"]) == 3
        assert all(s == 1 for s in metrics["exact_match_accuracy_scores"])

    def test_full_pipeline_partial_answers(self, sample_ask_queries):
        """Agent gets some answers wrong -> partial accuracy."""
        agent = MockAskAgent({
            "What is the capital of France?": "Paris",
            "What is 2+2?": "5",  # Wrong!
            "Who wrote Hamlet?": "Charles Dickens",  # Wrong!
        })

        results = run_ask_queries(sample_ask_queries, agent)
        calculator = ExactMatchAskCalculator()
        metrics = calculator.compute(results)

        # Only 1 out of 3 correct
        assert metrics["avg_exact_match_accuracy"] == pytest.approx(1 / 3)
        assert metrics["exact_match_accuracy_scores"] == [1, 0, 0]

    def test_full_pipeline_with_errors(self, sample_ask_queries):
        """Agent raises errors on some queries -> errors are handled gracefully."""

        class ErrorAgent:
            def run(self, query, oracle_context_id=None, tenant_id=None):
                if "2+2" in query:
                    raise ValueError("Simulated error")
                return AskResponse(final_answer="Paris")

        results = run_ask_queries(sample_ask_queries, ErrorAgent())
        assert len(results) == 3

        # Error result is tagged
        error_result = results[1]
        assert error_result.system_answer.startswith("[ERROR]")

        # Metrics calculator skips error results
        calculator = ExactMatchAskCalculator()
        metrics = calculator.compute(results)
        assert len(metrics["exact_match_accuracy_scores"]) == 2  # Only 2 evaluated

    def test_full_pipeline_aggregation(self, sample_ask_queries):
        """Multi-trial aggregation produces correct structure."""
        agent = MockAskAgent({
            "What is the capital of France?": "Paris",
            "What is 2+2?": "4",
            "Who wrote Hamlet?": "William Shakespeare",
        })

        calculator = ExactMatchAskCalculator()

        all_metrics = []
        for _ in range(3):
            results = run_ask_queries(sample_ask_queries, agent)
            metrics = calculator.compute(results)
            all_metrics.append(metrics)

        aggregated = aggregate_metrics(all_metrics)

        assert aggregated["num_trials"] == 3
        assert "avg_exact_match_accuracy_mean" in aggregated
        assert aggregated["avg_exact_match_accuracy_mean"] == pytest.approx(1.0)
        assert len(aggregated["trials"]) == 3

    def test_orchestrator_end_to_end(self, sample_ask_queries):
        """Test AskBenchmarkOrchestrator with all-mocked dependencies."""
        agent = MockAskAgent({
            "What is the capital of France?": "Paris",
            "What is 2+2?": "4",
            "Who wrote Hamlet?": "William Shakespeare",
        })

        calculator = ExactMatchAskCalculator()
        mock_repo = MagicMock()

        orchestrator = AskBenchmarkOrchestrator(
            query_runner=run_ask_queries,
            metrics_calculator=calculator,
            result_repository=mock_repo,
        )

        config = {"dataset_identifier": "test", "agent_name": "mock"}
        aggregated = orchestrator.run_and_aggregate(
            queries=sample_ask_queries,
            agent=agent,
            config=config,
            num_trials=2,
        )

        assert aggregated["num_trials"] == 2
        assert mock_repo.save_ask_trial_results.call_count == 2
        assert mock_repo.save_trial_metrics.call_count == 2
        assert mock_repo.save_aggregated_results.call_count == 1

    def test_tenant_propagation(self):
        """Tenant IDs are correctly propagated through the pipeline."""
        queries = [
            InMemoryAskQuery(
                question="What happened yesterday?",
                ground_truth_answer="Meeting",
                tenant_id="user-001",
            ),
        ]

        received_tenants = []

        class TenantCapturingAgent:
            def run(self, query, oracle_context_id=None, tenant_id=None):
                received_tenants.append(tenant_id)
                return AskResponse(final_answer="Meeting")

        results = run_ask_queries(queries, TenantCapturingAgent())
        assert received_tenants == ["user-001"]
        assert results[0].system_answer == "Meeting"

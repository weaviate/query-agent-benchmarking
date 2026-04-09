"""Tests for domain query execution functions."""

import pytest

from query_agent_benchmarking.internal.core.domain.query_execution import (
    run_search_queries,
    run_ask_queries,
)
from query_agent_benchmarking.internal.core.domain.models import (
    InMemoryQuery,
    InMemoryAskQuery,
    ObjectID,
)
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse


class MockSearchAgent:
    def __init__(self):
        self.queries_received = []

    def run(self, query, tenant=None):
        self.queries_received.append((query, tenant))
        return [ObjectID(object_id="doc1"), ObjectID(object_id="doc2")]


class MockAskAgent:
    def __init__(self):
        self.queries_received = []

    def run(self, query, oracle_context_id=None, tenant_id=None):
        self.queries_received.append((query, oracle_context_id, tenant_id))
        return AskResponse(final_answer="Test answer")


class TestRunSearchQueries:
    def test_basic_execution(self):
        queries = [
            InMemoryQuery(question="Q1", dataset_ids=["d1"]),
            InMemoryQuery(question="Q2", dataset_ids=["d2", "d3"]),
        ]
        agent = MockSearchAgent()
        results = run_search_queries(queries, agent)

        assert len(results) == 2
        assert len(agent.queries_received) == 2
        assert agent.queries_received[0] == ("Q1", None)
        assert results[0].query.question == "Q1"
        assert results[0].query_ground_truth_id == ["d1"]
        assert len(results[0].retrieved_ids) == 2
        assert results[0].time_taken > 0

    def test_tenant_propagation(self):
        queries = [
            InMemoryQuery(question="Q1", dataset_ids=["d1"], tenant_id="tenant-a"),
        ]
        agent = MockSearchAgent()
        run_search_queries(queries, agent)

        assert agent.queries_received[0] == ("Q1", "tenant-a")

    def test_empty_queries(self):
        results = run_search_queries([], MockSearchAgent())
        assert results == []


class TestRunAskQueries:
    def test_basic_execution(self):
        queries = [
            InMemoryAskQuery(question="What is X?", ground_truth_answer="X is Y."),
        ]
        agent = MockAskAgent()
        results = run_ask_queries(queries, agent)

        assert len(results) == 1
        assert results[0].system_answer == "Test answer"
        assert results[0].time_taken > 0

    def test_oracle_context_propagation(self):
        queries = [
            InMemoryAskQuery(
                question="Q1",
                ground_truth_answer="A1",
                oracle_context_id="ctx-1",
                tenant_id="t1",
            ),
        ]
        agent = MockAskAgent()
        run_ask_queries(queries, agent)

        assert agent.queries_received[0] == ("Q1", "ctx-1", "t1")

    def test_error_handling(self):
        class ErrorAgent:
            def run(self, query, oracle_context_id=None, tenant_id=None):
                raise RuntimeError("Test error")

        queries = [
            InMemoryAskQuery(question="Q1", ground_truth_answer="A1"),
        ]
        results = run_ask_queries(queries, ErrorAgent())

        assert len(results) == 1
        assert results[0].system_answer.startswith("[ERROR]")

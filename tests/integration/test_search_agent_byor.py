"""Tests for the bring-your-own-retriever (search_agent) parameter.

Verifies that users can pass a custom SearchAgent instance to run_search_eval,
bypassing the agent_name factory lookup.
"""

import asyncio
import pytest
from unittest.mock import patch

from query_agent_benchmarking.internal.core.domain.models import (
    InMemoryQuery,
    ObjectID,
)
from query_agent_benchmarking.internal.core.ports.search_agent import SearchAgent
from query_agent_benchmarking.internal.core.services.search_benchmark import (
    _run_search_eval,
)
from query_agent_benchmarking.internal.mocks.agents import MockSearchAgent


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
    ]


@pytest.fixture
def mock_agent():
    return MockSearchAgent(responses={
        "What is information retrieval?": ["doc1", "doc2"],
        "How does BM25 work?": ["doc3"],
    })


def _build_config(queries, agent, **overrides):
    """Build a minimal config dict for _run_search_eval with a user-provided agent."""
    config = {
        "search_dataset": "beir/scifact",
        "search_agent": agent,
        "use_async": False,
        "num_trials": 1,
    }
    config.update(overrides)
    return config


# ============================================================================
# Tests
# ============================================================================

class TestSearchAgentBYOR:
    """Test the bring-your-own-retriever path through run_search_eval."""

    def test_user_provided_agent_is_used(self, sample_queries, mock_agent):
        """When search_agent is provided, it should be used instead of the factory."""
        config = _build_config(sample_queries, mock_agent)

        with patch(
            "query_agent_benchmarking.internal.core.services.search_benchmark.create_search_agent"
        ) as mock_factory:
            result = asyncio.run(_run_search_eval(config))
            mock_factory.assert_not_called()

        assert mock_agent.call_count == len(sample_queries)

    def test_user_provided_agent_metrics_computed(self, sample_queries, mock_agent):
        """Metrics should be computed normally with a user-provided agent."""
        config = _build_config(sample_queries, mock_agent)

        result = asyncio.run(_run_search_eval(config))

        # Perfect retrieval -> recall@20 should be 1.0
        assert result["avg_recall_at_20"] == pytest.approx(1.0)

    def test_agent_name_defaults_to_class_name(self, sample_queries, mock_agent):
        """When no agent_name is in config, should fall back to the class name."""
        config = _build_config(sample_queries, mock_agent)
        # Ensure no agent_name is set
        config.pop("agent_name", None)
        config.pop("search_agent_name", None)

        asyncio.run(_run_search_eval(config))

        # After _run_search_eval, config should have the class name
        assert config["agent_name"] == "MockSearchAgent"

    def test_explicit_agent_name_preserved(self, sample_queries, mock_agent):
        """When agent_name is provided alongside search_agent, it should be used."""
        config = _build_config(
            sample_queries, mock_agent, agent_name="my-custom-retriever"
        )

        asyncio.run(_run_search_eval(config))

        assert config["agent_name"] == "my-custom-retriever"

    def test_non_protocol_object_raises_type_error(self, sample_queries):
        """Passing an object that doesn't implement SearchAgent should raise TypeError."""

        class NotASearchAgent:
            pass

        config = _build_config(sample_queries, NotASearchAgent())

        with pytest.raises(TypeError, match="SearchAgent protocol"):
            asyncio.run(_run_search_eval(config))

    def test_async_execution_with_user_agent(self, sample_queries, mock_agent):
        """User-provided agent should work with async execution path."""
        config = _build_config(sample_queries, mock_agent, use_async=True)

        with patch(
            "query_agent_benchmarking.internal.core.services.search_benchmark.create_search_agent"
        ) as mock_factory:
            result = asyncio.run(_run_search_eval(config))
            mock_factory.assert_not_called()

        assert mock_agent.call_count == len(sample_queries)
        assert result["avg_recall_at_20"] == pytest.approx(1.0)

    def test_multi_trial_with_user_agent(self, sample_queries, mock_agent):
        """User-provided agent should work across multiple trials."""
        config = _build_config(sample_queries, mock_agent, num_trials=3)

        result = asyncio.run(_run_search_eval(config))

        assert result["num_trials"] == 3
        assert mock_agent.call_count == len(sample_queries) * 3

    def test_protocol_compliance_check(self):
        """MockSearchAgent should pass the runtime_checkable SearchAgent protocol."""
        agent = MockSearchAgent()
        assert isinstance(agent, SearchAgent)

    def test_public_api_accepts_search_agent(self, sample_queries, mock_agent):
        """run_search_eval (public API) should accept and forward search_agent."""
        from query_agent_benchmarking.internal.core.services.search_benchmark import (
            run_search_eval,
        )

        with patch(
            "query_agent_benchmarking.internal.core.services.search_benchmark.create_search_agent"
        ) as mock_factory:
            result = run_search_eval(
                search_dataset="beir/scifact",
                search_agent=mock_agent,
                use_async=False,
                num_trials=1,
            )
            mock_factory.assert_not_called()

        assert result["avg_recall_at_20"] == pytest.approx(1.0)

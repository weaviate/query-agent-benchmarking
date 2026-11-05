"""End-to-end test of QueryAgentGym search mode through ``run_search_eval``.

The dataset loader, agent factory and result repository are patched, so this
runs offline while exercising the service wiring: per-environment collections,
UUID results scored with nDCG, plan scoring with query recall, and per-query
detail persistence.
"""

import pytest

from query_agent_benchmarking.internal.core.domain.models import (
    AgentSearch,
    InMemoryQuery,
    ObjectID,
    SearchAgentResponse,
)
from query_agent_benchmarking.internal.core.services import search_benchmark
from query_agent_benchmarking.internal.mocks import MockResultRepository


DATASET = "query-agent-gym-search-mode"
COLLECTIONS = {
    "childcare": {"DaycareCenters_Default": "DaycareCenters"},
    "surf-schools": {"SurfBeaches_Default": "SurfBeaches"},
}

QUERIES = [
    InMemoryQuery(
        question="Which daycare centers provide meals?",
        dataset_ids=["u1", "u2"],
        query_id="childcare-020",
        ground_truth_queries=[{
            "collection_name": "DaycareCenters",
            "boolean_property_filters": [{"property_name": "providesMeals", "operator": "=", "value": True}],
        }],
        metadata={"environment": "childcare", "retrieval_shape": "filter_only", "operator_classes": ["boolean filter"]},
    ),
    InMemoryQuery(
        question="Which beaches suit sheltered morning lessons?",
        dataset_ids=["u3", "u4"],
        query_id="surf-schools-000",
        ground_truth_queries=[{"collection_name": "SurfBeaches", "search_query": "sheltered morning lessons"}],
        metadata={"environment": "surf-schools", "retrieval_shape": "search_only", "operator_classes": ["search query"]},
    ),
]

RESPONSES = {
    QUERIES[0].question: SearchAgentResponse(
        retrieved_ids=[ObjectID(object_id="u1"), ObjectID(object_id="u2")],
        searches=[AgentSearch(
            collection="DaycareCenters_Default",
            filters={"filter_type": "boolean", "property_name": "providesMeals", "operator": "=", "value": True},
        )],
    ),
    # Searched (right plan) but returned nothing relevant.
    QUERIES[1].question: SearchAgentResponse(
        retrieved_ids=[ObjectID(object_id="x")],
        searches=[AgentSearch(collection="SurfBeaches_Default", query="sheltered beaches for morning lessons")],
    ),
}


class PlanSearchAgent:
    def __init__(self):
        self.collections = None
        self.collections_by_question = {}

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass

    async def run_async(self, query, tenant=None, collections=None):
        self.collections_by_question[query] = collections
        return RESPONSES[query]


@pytest.fixture
def patched(monkeypatch):
    agent = PlanSearchAgent()
    repo = MockResultRepository()

    def fake_create_search_agent(agent_name, *, collections=None, **kwargs):
        agent.collections = collections
        return agent

    monkeypatch.setattr(
        search_benchmark, "in_memory_dataset_loader",
        lambda name, queries_only=False: ([], [q.model_copy() for q in QUERIES]),
    )
    monkeypatch.setattr(search_benchmark, "query_agent_gym_collection_names", lambda name: dict(COLLECTIONS))
    monkeypatch.setattr(search_benchmark, "query_agent_gym_array_properties", lambda name: {})
    monkeypatch.setattr(search_benchmark, "query_agent_gym_date_ranges", lambda name: {})
    monkeypatch.setattr(search_benchmark, "create_search_agent", fake_create_search_agent)
    monkeypatch.setattr(search_benchmark, "JsonFileResultRepository", lambda: repo)
    return agent, repo


def test_query_agent_gym_search_eval(patched, tmp_path):
    agent, repo = patched
    aggregated = search_benchmark.run_search_eval(
        search_dataset=DATASET,
        config_path=str(tmp_path / "missing.yml"),
        agent_name="query-agent-search-mode",
        num_trials=1,
        use_async=True,
    )

    # The agent is built over every environment, but each query sees only its own.
    assert agent.collections == ["DaycareCenters_Default", "SurfBeaches_Default"]
    assert agent.collections_by_question == {
        QUERIES[0].question: ["DaycareCenters_Default"],
        QUERIES[1].question: ["SurfBeaches_Default"],
    }
    assert aggregated["avg_query_recall_mean"] == 1.0
    assert aggregated["avg_nDCG_at_10_mean"] == 0.5
    assert aggregated["retrieval_shape_nDCG_at_10"]["filter_only"]["mean"] == 1.0
    assert aggregated["retrieval_shape_nDCG_at_10"]["search_only"]["mean"] == 0.0
    assert aggregated["environment_accuracy"]["surf-schools"]["mean"] == 1.0

    trial_metrics = repo.saved_metrics[0][0]
    assert "query_match_details" not in trial_metrics


def test_other_search_agents_are_rejected(patched, tmp_path):
    with pytest.raises(ValueError, match="query-agent-search-mode"):
        search_benchmark.run_search_eval(
            search_dataset=DATASET,
            config_path=str(tmp_path / "missing.yml"),
            agent_name="hybrid-search",
            num_trials=1,
        )

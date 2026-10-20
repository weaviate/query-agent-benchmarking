"""End-to-end test of the QueryAgentGym ask flow through ``run_ask_eval``.

The dataset loader, agent factory and result repository are patched, so this
runs offline while exercising the service wiring: multi-collection agent
construction, query exact-match scoring, and per-query detail persistence.
"""

import pytest

from query_agent_benchmarking.internal.core.domain.models import AgentSearch, InMemoryAskQuery
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse
from query_agent_benchmarking.internal.core.services import ask_benchmark
from query_agent_benchmarking.internal.mocks import MockResultRepository


DATASET = "query-agent-gym/childcare"
ALIASES = {"DaycareCenters_Default": "DaycareCenters", "ChildcareStaff_Default": "ChildcareStaff"}

QUERIES = [
    InMemoryAskQuery(
        question="Which daycare centers provide meals?",
        ground_truth_answer="SELECT * ...",
        query_id="childcare-900",
        ground_truth_queries=[{
            "collection_name": "DaycareCenters",
            "boolean_property_filters": [{"property_name": "providesMeals", "operator": "=", "value": True}],
        }],
        metadata={"operator_classes": ["boolean filter"], "phrasing": "explicit"},
    ),
    InMemoryAskQuery(
        question="Show caregivers named Odelia.",
        ground_truth_answer="SELECT * ...",
        query_id="childcare-901",
        ground_truth_queries=[{
            "collection_name": "ChildcareStaff",
            "text_property_filters": [{"property_name": "staffName", "operator": "LIKE", "value": "Odelia%"}],
        }],
        metadata={"operator_classes": ["text filter"], "phrasing": "conversational"},
    ),
]

PLANS = {
    QUERIES[0].question: [AgentSearch(
        collection="DaycareCenters_Default",
        filters={"filter_type": "boolean", "property_name": "providesMeals", "operator": "=", "value": True},
    )],
    # Wrong: an exact match instead of a prefix match.
    QUERIES[1].question: [AgentSearch(
        collection="ChildcareStaff_Default",
        filters={"filter_type": "text", "property_name": "staffName", "operator": "=", "value": "Odelia"},
    )],
}


class PlanAskAgent:
    def __init__(self):
        self.collections = None

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass

    async def run_async(self, query, oracle_context_id=None, tenant_id=None):
        return AskResponse(final_answer="answer", searches=PLANS[query], aggregations=[])


@pytest.fixture
def patched(monkeypatch):
    agent = PlanAskAgent()
    repo = MockResultRepository()

    def fake_create_ask_agent(agent_name, *, collections=None, **kwargs):
        agent.collections = collections
        return agent

    monkeypatch.setattr(ask_benchmark, "in_memory_ask_dataset_loader", lambda name, **kw: list(QUERIES))
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_collection_names", lambda name: dict(ALIASES))
    monkeypatch.setattr(ask_benchmark, "create_ask_agent", fake_create_ask_agent)
    monkeypatch.setattr(ask_benchmark, "JsonFileResultRepository", lambda: repo)
    return agent, repo


def test_query_agent_gym_eval(patched, tmp_path):
    agent, repo = patched
    aggregated = ask_benchmark.run_ask_eval(
        queries=DATASET,
        config_path=str(tmp_path / "missing.yml"),
        agent_name="query-agent-ask-mode",
        num_trials=1,
        use_async=True,
    )

    assert agent.collections == list(ALIASES)
    assert aggregated["avg_query_exact_match_mean"] == 0.5
    assert aggregated["type_accuracy"]["boolean filter"]["mean"] == 1.0
    assert aggregated["type_accuracy"]["text filter"]["mean"] == 0.0
    assert "avg_alignment_score_mean" not in aggregated

    results, _, _, scores, _ = repo.saved_ask_trials[0]
    assert scores == [1, 0]
    assert results[0].searches == PLANS[QUERIES[0].question]

    trial_metrics = repo.saved_metrics[0][0]
    assert "query_match_details" not in trial_metrics

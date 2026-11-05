"""End-to-end test of the QueryAgentGym ask flow through ``run_ask_eval``.

The dataset loader, agent factory and result repository are patched, so this
runs offline while exercising the service wiring: per-environment collections
across the full multi-environment dataset, query exact-match scoring, and
per-query detail persistence.
"""

import pytest

from query_agent_benchmarking.internal.core.domain.models import AgentSearch, InMemoryAskQuery
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse
from query_agent_benchmarking.internal.core.services import ask_benchmark
from query_agent_benchmarking.internal.mocks import MockResultRepository


DATASET = "query-agent-gym"
COLLECTIONS = {
    "childcare": {"DaycareCenters_Default": "DaycareCenters", "ChildcareStaff_Default": "ChildcareStaff"},
    "surf-schools": {"SurfBeaches_Default": "SurfBeaches"},
}

QUERIES = [
    InMemoryAskQuery(
        question="Which daycare centers provide meals?",
        ground_truth_answer="SELECT * ...",
        query_id="childcare-900",
        ground_truth_queries=[{
            "collection_name": "DaycareCenters",
            "boolean_property_filters": [{"property_name": "providesMeals", "operator": "=", "value": True}],
        }],
        metadata={"environment": "childcare", "operator_classes": ["boolean filter"], "phrasing": "explicit"},
    ),
    InMemoryAskQuery(
        question="Show caregivers named Odelia.",
        ground_truth_answer="SELECT * ...",
        query_id="childcare-901",
        ground_truth_queries=[{
            "collection_name": "ChildcareStaff",
            "text_property_filters": [{"property_name": "staffName", "operator": "LIKE", "value": "Odelia%"}],
        }],
        metadata={"environment": "childcare", "operator_classes": ["text filter"], "phrasing": "conversational"},
    ),
    InMemoryAskQuery(
        question="Which beaches are good for beginners?",
        ground_truth_answer="SELECT * ...",
        query_id="surf-schools-900",
        ground_truth_queries=[{
            "collection_name": "SurfBeaches",
            "boolean_property_filters": [{"property_name": "beginnerFriendly", "operator": "=", "value": True}],
        }],
        metadata={"environment": "surf-schools", "operator_classes": ["boolean filter"], "phrasing": "explicit"},
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
    QUERIES[2].question: [AgentSearch(
        collection="SurfBeaches_Default",
        filters={"filter_type": "boolean", "property_name": "beginnerFriendly", "operator": "=", "value": True},
    )],
}


class PlanAskAgent:
    def __init__(self):
        self.collections = None
        self.collections_by_question = {}

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass

    async def run_async(self, query, oracle_context_id=None, tenant_id=None, collections=None):
        self.collections_by_question[query] = collections
        return AskResponse(final_answer="answer", searches=PLANS[query], aggregations=[])


@pytest.fixture
def patched(monkeypatch):
    agent = PlanAskAgent()
    repo = MockResultRepository()

    def fake_create_ask_agent(agent_name, *, collections=None, **kwargs):
        agent.collections = collections
        return agent

    monkeypatch.setattr(
        ask_benchmark, "in_memory_ask_dataset_loader", lambda name, **kw: [q.model_copy() for q in QUERIES]
    )
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_collection_names", lambda name: dict(COLLECTIONS))
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_array_properties", lambda name: {})
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_date_ranges", lambda name: {})
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

    # The agent is built over every environment, but each question sees only its own.
    assert agent.collections == [*COLLECTIONS["childcare"], *COLLECTIONS["surf-schools"]]
    assert agent.collections_by_question == {
        QUERIES[0].question: list(COLLECTIONS["childcare"]),
        QUERIES[1].question: list(COLLECTIONS["childcare"]),
        QUERIES[2].question: list(COLLECTIONS["surf-schools"]),
    }
    assert aggregated["avg_query_recall_mean"] == pytest.approx(2 / 3)
    assert aggregated["avg_query_exact_match_mean"] == pytest.approx(2 / 3)
    assert aggregated["type_accuracy"]["boolean filter"]["mean"] == 1.0
    assert aggregated["type_accuracy"]["text filter"]["mean"] == 0.0
    assert aggregated["environment_accuracy"]["childcare"]["mean"] == 0.5
    assert aggregated["environment_accuracy"]["surf-schools"]["mean"] == 1.0
    assert "avg_alignment_score_mean" not in aggregated

    results, _, _, scores, _ = repo.saved_ask_trials[0]
    assert scores == [1, 0, 1]
    assert results[0].searches == PLANS[QUERIES[0].question]

    trial_metrics = repo.saved_metrics[0][0]
    assert "query_match_details" not in trial_metrics


EDGE_CASES = [
    InMemoryAskQuery(
        question='Show me the daycare center named "Mirevale Commons".',
        ground_truth_answer="B1",
        query_id="childcare-b1-000",
        metadata={
            "environment": "childcare", "behavior": "B1", "node": "query", "collection": "DaycareCenters",
            "phrasing": "quoted", "value": "Mirevale Commons",
            "expect": {"value_filter": "eq", "value_properties": ["centerName"], "query_contains": [],
                       "query_null": True, "other_filters": []},
        },
    ),
    InMemoryAskQuery(
        question="which beaches relate to driftwood?",
        ground_truth_answer="B9",
        query_id="surf-schools-b9-000",
        metadata={
            "environment": "surf-schools", "behavior": "B9", "node": "query", "collection": "SurfBeaches",
            "phrasing": "casual", "value": "driftwood",
            "expect": {"value_filter": "no_eq", "value_properties": [], "query_contains": [],
                       "query_null": False, "other_filters": []},
        },
    ),
]

EDGE_PLANS = {
    EDGE_CASES[0].question: [AgentSearch(
        collection="DaycareCenters_Default",
        filters={"filter_type": "text", "property_name": "centerName", "operator": "=", "value": "Mirevale Commons"},
    )],
    # Wrong: an exact-match filter on a vague term.
    EDGE_CASES[1].question: [AgentSearch(
        collection="SurfBeaches_Default",
        filters={"filter_type": "text", "property_name": "beachName", "operator": "=", "value": "driftwood"},
    )],
}


def test_query_agent_gym_edge_case_eval(monkeypatch, tmp_path):
    agent = PlanAskAgent()
    repo = MockResultRepository()

    async def run_async(query, oracle_context_id=None, tenant_id=None, collections=None):
        agent.collections_by_question[query] = collections
        return AskResponse(final_answer="answer", searches=EDGE_PLANS[query], aggregations=[])

    agent.run_async = run_async
    monkeypatch.setattr(
        ask_benchmark, "in_memory_ask_dataset_loader", lambda name, **kw: [q.model_copy() for q in EDGE_CASES]
    )
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_collection_names", lambda name: dict(COLLECTIONS))
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_array_properties", lambda name: {})
    monkeypatch.setattr(ask_benchmark, "query_agent_gym_date_ranges", lambda name: {})
    monkeypatch.setattr(ask_benchmark, "create_ask_agent", lambda agent_name, **kwargs: agent)
    monkeypatch.setattr(ask_benchmark, "JsonFileResultRepository", lambda: repo)

    aggregated = ask_benchmark.run_ask_eval(
        queries="query-agent-gym-edge-case-1",
        config_path=str(tmp_path / "missing.yml"),
        agent_name="query-agent-ask-mode",
        num_trials=1,
        use_async=True,
    )

    assert agent.collections_by_question[EDGE_CASES[1].question] == list(COLLECTIONS["surf-schools"])
    assert aggregated["avg_edge_case_pass_mean"] == 0.5
    assert aggregated["type_accuracy"]["B1"]["mean"] == 1.0
    assert aggregated["type_accuracy"]["B9"]["mean"] == 0.0
    _, _, _, scores, _ = repo.saved_ask_trials[0]
    assert scores == [1, 0]

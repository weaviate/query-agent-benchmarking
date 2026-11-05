"""Tests for QueryAgentGym search mode: the results-and-plan calculator, the
Query Agent search adapter's multi-collection / UUID mode, and per-query
collections in search execution. All offline."""

from types import SimpleNamespace

import pytest

from query_agent_benchmarking.internal.adapters.agents.weaviate_query_agent import WeaviateQueryAgentSearch
from query_agent_benchmarking.internal.adapters.metrics.query_agent_gym_search_calculator import (
    QueryAgentGymSearchCalculator,
)
from query_agent_benchmarking.internal.core.domain.models import (
    AgentSearch,
    InMemoryQuery,
    ObjectID,
    QueryResult,
)
from query_agent_benchmarking.internal.core.domain.query_execution import run_search_queries


DATASET = "query-agent-gym/search-mode"
ALIASES = {"Centers_Default": "Centers"}
MEALS_FILTER = {"filter_type": "boolean", "property_name": "meals", "operator": "=", "value": True}


def _query(shape, ground_truth, relevant, environment="childcare"):
    return InMemoryQuery(
        question=f"{shape} question",
        dataset_ids=relevant,
        query_id=f"{environment}-{shape}",
        ground_truth_queries=ground_truth,
        metadata={"environment": environment, "retrieval_shape": shape, "operator_classes": [shape]},
    )


def _result(query, retrieved, searches, error=None):
    return QueryResult(
        query=query,
        query_ground_truth_id=query.dataset_ids,
        retrieved_ids=[ObjectID(object_id=i) for i in retrieved],
        time_taken=1.0,
        searches=searches,
        error=error,
    )


FILTER_GT = [{"collection_name": "Centers", "boolean_property_filters": [
    {"property_name": "meals", "operator": "=", "value": True}]}]
SEARCH_GT = [{"collection_name": "Centers", "search_query": "garden play"}]


class TestQueryAgentGymSearchCalculator:
    def test_scores_results_and_plans_by_shape(self):
        filter_query = _query("filter_only", FILTER_GT, ["a", "b"])
        search_query = _query("search_only", SEARCH_GT, ["c", "d"])
        failed_query = _query("search_only", SEARCH_GT, ["e"])
        results = [
            # Right plan plus an extra search: query recall 1, exact match 0. Perfect results.
            _result(filter_query, ["a", "b"], [
                AgentSearch(collection="Centers_Default", filters=MEALS_FILTER),
                AgentSearch(collection="Centers_Default", query="meals"),
            ]),
            # Wrong plan (a filter instead of a search); one relevant result, ranked second.
            _result(search_query, ["x", "c"], [AgentSearch(collection="Centers_Default", filters=MEALS_FILTER)]),
            _result(failed_query, [], None, error="boom"),
        ]
        queries = [r.query for r in results]
        metrics = QueryAgentGymSearchCalculator(DATASET, collection_aliases=ALIASES).compute(results, queries)

        assert metrics["metric"] == "query_recall"
        assert metrics["num_errors"] == 1
        assert metrics["query_recall_scores"] == [1, 0, None]
        assert metrics["avg_query_recall"] == 0.5
        assert metrics["query_exact_match_scores"] == [0, 0, None]
        assert metrics["retrieval_shape_accuracy"] == {"filter_only": 1.0, "search_only": 0.0}
        assert metrics["nDCG_at_10_scores"][0] == 1.0
        assert 0 < metrics["nDCG_at_10_scores"][1] < 1
        assert metrics["nDCG_at_10_scores"][2] is None
        # The failed query is left out of the IR averages too.
        assert metrics["avg_nDCG_at_10"] == pytest.approx((1.0 + metrics["nDCG_at_10_scores"][1]) / 2)
        assert metrics["retrieval_shape_nDCG_at_10"]["filter_only"] == 1.0

        details = metrics["query_match_details"]
        assert details[2] is None
        assert details[0]["extra_queries"] == ["SELECT * FROM Centers SEARCH <query>"]
        assert details[1]["missing_queries"] == ["SELECT * FROM Centers SEARCH <query>"]
        assert details[0]["nDCG_at_10"] == 1.0

    def test_requires_ground_truth_queries(self):
        query = InMemoryQuery(question="q", dataset_ids=["a"])
        with pytest.raises(ValueError, match="ground_truth_queries"):
            QueryAgentGymSearchCalculator(DATASET).compute([_result(query, ["a"], [])], [query])


class TestWeaviateQueryAgentSearchCollections:
    def test_results_are_uuids_and_collections_are_per_query(self):
        calls = []

        class FakeQueryAgent:
            def search(self, query, **kwargs):
                calls.append(kwargs.get("collections"))
                objects = [SimpleNamespace(uuid="u-1", properties={}), SimpleNamespace(uuid="u-2", properties={})]
                return SimpleNamespace(search_results=SimpleNamespace(objects=objects), searches=[])

        agent = WeaviateQueryAgentSearch(collections=["A_Default", "B_Default"])
        assert agent.collections == ["A_Default", "B_Default"]
        agent._agent = FakeQueryAgent()

        response = agent.run("q", collections=["A_Default"])
        agent.run("q")

        assert [o.object_id for o in response.retrieved_ids] == ["u-1", "u-2"]
        assert calls == [["A_Default"], None]


class TestSearchExecutionCollections:
    def test_collections_only_passed_when_the_query_sets_them(self):
        class LegacyAgent:
            """A search agent that predates the ``collections`` argument."""
            def run(self, query, tenant=None):
                return [ObjectID(object_id="a")]

        class CollectionsAgent:
            def __init__(self):
                self.seen = []

            def run(self, query, tenant=None, collections=None):
                self.seen.append(collections)
                return [ObjectID(object_id="a")]

        plain = InMemoryQuery(question="q", dataset_ids=["a"])
        scoped = InMemoryQuery(question="q", dataset_ids=["a"], collections=["A_Default"])

        assert len(run_search_queries([plain], LegacyAgent())) == 1
        agent = CollectionsAgent()
        run_search_queries([plain, scoped], agent)
        assert agent.seen == [None, ["A_Default"]]

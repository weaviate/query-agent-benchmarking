"""Tests for QueryAgentGym's exact-value filtering checks (edge_case_1). All offline."""

import pytest

from query_agent_benchmarking.internal.adapters.metrics.query_agent_gym_edge_case import (
    QueryAgentGymEdgeCaseAskCalculator,
    check_edge_case,
)
from query_agent_benchmarking.internal.core.domain.models import (
    AgentAggregation,
    AgentSearch,
    AskResult,
    InMemoryAskQuery,
)


ALIASES = {"Reports_Default": "Reports"}
DATE_RANGES = {"Reports": {"serviceDate": ("2025-01-15", "2026-09-18")}}


def _case(behavior, value, node="query", **expect):
    return {
        "environment": "public-transit", "behavior": behavior, "node": node,
        "collection": "Reports", "phrasing": "quoted", "value": value,
        "expect": {
            "value_filter": "eq", "value_properties": [], "query_contains": [],
            "query_null": False, "other_filters": [], **expect,
        },
    }


def _text(prop, op, value):
    return {"filter_type": "text", "property_name": prop, "operator": op, "value": value}


def _search(query=None, filters=None):
    return AgentSearch(collection="Reports_Default", query=query, filters=filters)


def _check(case, searches=(), aggregations=()):
    return check_edge_case(case, list(searches), list(aggregations), ALIASES, {}, DATE_RANGES)


B1 = _case("B1", "RID-00469", value_properties=["reportCode"], query_null=True)


class TestCheckEdgeCase:
    def test_eq_filter_with_null_query_passes(self):
        assert _check(B1, [_search(filters=_text("reportCode", "=", "RID-00469"))])["passed"]

    def test_searching_for_the_value_fails(self):
        result = _check(B1, [_search(query="RID-00469")])
        assert not result["passed"]
        assert result["failures"] == ["value_filter", "query_null", "same_query"]

    def test_filter_plus_search_query_fails_when_query_must_be_null(self):
        result = _check(B1, [_search(query="report", filters=_text("reportCode", "=", "RID-00469"))])
        assert result["failures"] == ["query_null", "same_query"]

    def test_wrong_collection_fails(self):
        search = AgentSearch(collection="Routes_Default", filters=_text("reportCode", "=", "RID-00469"))
        assert "collection" in _check(B1, [search])["failures"]

    def test_extra_queries_are_allowed_for_positive_checks(self):
        searches = [_search(query="RID-00469"), _search(filters=_text("reportCode", "=", "RID-00469"))]
        assert _check(B1, searches)["passed"]

    def test_none_needs_the_value_in_the_query_and_never_in_a_filter(self):
        case = _case("B2", "Nacrecoil West Reach", value_filter="none", query_contains=["Nacrecoil West Reach"])
        assert _check(case, [_search(query="Nacrecoil West Reach")])["passed"]
        # A filter carrying the value fails even alongside a correct search.
        searches = [_search(query="Nacrecoil West Reach"),
                    _search(filters=_text("routeName", "LIKE", "*Nacrecoil West Reach*"))]
        assert _check(case, searches)["failures"] == ["value_not_filtered"]

    def test_like_prefix(self):
        case = _case("B4", "Nettle", value_filter="like_prefix", value_properties=["routeName"])
        assert _check(case, [_search(filters=_text("routeName", "LIKE", "Nettle%"))])["passed"]
        assert not _check(case, [_search(filters=_text("routeName", "=", "Nettle"))])["passed"]

    def test_other_filters_must_be_anded_in_the_same_query(self):
        case = _case("B5", "RID-00706", value_properties=["reportCode"], query_null=True, other_filters=[
            {"property_name": "serviceDate", "operator": ">", "value": "2025-06-01"}])
        date = {"filter_type": "date_range", "property_name": "serviceDate",
                "value": {"date_from": "2025-06-02T00:00:00Z", "inclusive_from": True}}
        together = {"combine": "AND", "filters": [_text("reportCode", "=", "RID-00706"), date]}
        assert _check(case, [_search(filters=together)])["passed"]
        apart = [_search(filters=_text("reportCode", "=", "RID-00706")), _search(filters=date)]
        assert _check(case, apart)["failures"] == ["same_query"]

    def test_or_eq_on_an_aggregation(self):
        case = _case("B8", "Quenlow Link", node="aggregate", value_filter="or_eq",
                     value_properties=["routeName", "comparisonRouteName"], query_null=True)
        count = {"property_name": "reportCode", "metrics": "COUNT"}
        both = {"combine": "OR", "filters": [
            _text("routeName", "=", "Quenlow Link"), _text("comparisonRouteName", "=", "Quenlow Link")]}
        assert _check(case, aggregations=[AgentAggregation(collection="Reports_Default", aggregation=count,
                                                           filters=both)])["passed"]
        one = AgentAggregation(collection="Reports_Default", aggregation=count,
                               filters=_text("routeName", "=", "Quenlow Link"))
        assert not _check(case, aggregations=[one])["passed"]

    @pytest.mark.parametrize("node,searches,aggregations,passed", [
        ("query", [_search(query="commons")], [], True),
        ("query", [_search(filters=_text("routeName", "=", "commons"))], [], False),
        ("aggregate", [], [AgentAggregation(collection="Reports_Default", aggregation={"metrics": "COUNT"},
                                            filters=_text("notes", "LIKE", "*commons*"))], True),
        # A multi-word LIKE is not a single-keyword LIKE.
        ("aggregate", [], [AgentAggregation(collection="Reports_Default", aggregation={"metrics": "COUNT"},
                                            filters=_text("notes", "LIKE", "*town commons*"))], False),
    ])
    def test_no_eq_for_vague_terms(self, node, searches, aggregations, passed):
        case = _case("B9", "commons", node=node, value_filter="no_eq")
        assert _check(case, searches, aggregations)["passed"] is passed


class TestEdgeCaseCalculator:
    def test_scores_and_breakdowns(self):
        b2 = _case("B2", "Nacrecoil West Reach", value_filter="none", query_contains=["Nacrecoil West Reach"])
        results = [
            AskResult(query=InMemoryAskQuery(question="q", ground_truth_answer="", metadata=B1),
                      system_answer="a", time_taken=1.0,
                      searches=[_search(filters=_text("reportCode", "=", "RID-00469"))], aggregations=[]),
            AskResult(query=InMemoryAskQuery(question="q", ground_truth_answer="", metadata=b2),
                      system_answer="a", time_taken=1.0,
                      searches=[_search(filters=_text("routeName", "=", "Nacrecoil West Reach"))], aggregations=[]),
            AskResult(query=InMemoryAskQuery(question="q", ground_truth_answer="", metadata=B1),
                      system_answer="[ERROR] boom", time_taken=1.0),
        ]
        metrics = QueryAgentGymEdgeCaseAskCalculator(ALIASES, date_ranges=DATE_RANGES).compute(results)

        assert metrics["metric"] == "edge_case_pass"
        assert metrics["edge_case_pass_scores"] == [1, 0, None]
        assert metrics["avg_edge_case_pass"] == 0.5
        assert metrics["type_accuracy"] == {"B1": 1.0, "B2": 0.0}
        assert metrics["check_failure_counts"] == {"query_contains": 1, "same_query": 1, "value_not_filtered": 1}
        details = metrics["query_match_details"]
        assert details[2] is None
        assert details[1]["issued_queries"] == ["SELECT * FROM Reports WHERE routeName = 'Nacrecoil West Reach'"]

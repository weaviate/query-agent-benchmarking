"""Tests for QueryAgentGym: query exact match, the ask adapter's plan extraction,
and schema-driven collection specs. All offline.

Predicted queries are built from the real ``weaviate_agents`` response models
so the tests track the SDK's actual serialized shapes.
"""

import pytest
import weaviate.collections.classes.config as wvcc
from weaviate_agents.query.classes import response as R

from query_agent_benchmarking.internal.adapters.agents.weaviate_query_agent import WeaviateQueryAgentAsk
from query_agent_benchmarking.internal.adapters.database import query_agent_gym as qag_db
from query_agent_benchmarking.internal.adapters.metrics.ask_metrics_calculator import (
    QueryExactMatchAskCalculator,
)
from query_agent_benchmarking.internal.adapters.metrics.query_exact_match import (
    canonicalize_ground_truth,
    canonicalize_predicted,
    score_query_units,
)
from query_agent_benchmarking.internal.core.domain.models import (
    AgentAggregation,
    AgentSearch,
    AskResult,
    InMemoryAskQuery,
)


ALIASES = {"Staff_Default": "Staff", "Centers_Default": "Centers"}


def _search(query=None, filters=None, collection="Staff_Default", **kwargs) -> AgentSearch:
    sdk = R.QueryResultWithCollectionNormalized(query=query, filters=filters, collection=collection, **kwargs)
    return AgentSearch(
        collection=sdk.collection,
        query=sdk.query,
        filters=sdk.filters.model_dump(mode="json") if sdk.filters else None,
        uuid_value=str(sdk.uuid_value) if sdk.uuid_value else None,
    )


def _aggregation(aggregation, filters=None, groupby=None, collection="Staff_Default") -> AgentAggregation:
    sdk = R.AggregationResultWithCollectionNormalized(
        groupby_property=groupby, aggregation=aggregation, filters=filters, collection=collection,
    )
    return AgentAggregation(
        collection=sdk.collection,
        aggregation=sdk.aggregation.model_dump(mode="json"),
        filters=sdk.filters.model_dump(mode="json") if sdk.filters else None,
        groupby_property=sdk.groupby_property,
    )


def _match(ground_truth, searches=(), aggregations=()):
    gt = canonicalize_ground_truth(ground_truth)
    pred = canonicalize_predicted(list(searches), list(aggregations), ALIASES)
    return score_query_units(gt, pred)


# ============================================================================
# Query exact match
# ============================================================================


class TestQueryExactMatch:
    def test_multi_filter_retrieval_with_equivalent_forms(self):
        gt = [{
            "collection_name": "Staff",
            "text_property_filters": [{"property_name": "name", "operator": "LIKE", "value": "Odelia%"}],
            "numeric_array_property_filters": [
                {"property_name": "hours", "operator": "CONTAINS_ANY", "values": [40]}],
            "boolean_array_property_filters": [
                {"property_name": "checks", "operator": "CONTAINS_ANY", "values": [False]}],
        }]
        search = _search(filters=R.FilterAndOr(combine="AND", filters=[
            R.TextPropertyFilter(property_name="name", operator="LIKE", value="Odelia*"),
            R.FilterAndOr(combine="AND", filters=[
                R.IntegerArrayPropertyFilter(property_name="hours", operator="contains_all", value=[40]),
                R.BooleanArrayPropertyFilter(property_name="checks", operator="=", value=[False]),
            ]),
        ]))
        assert _match(gt, [search])["exact_match"]

    def test_search_text_is_not_compared_but_presence_is(self):
        gt = [{"collection_name": "Staff", "search_query": "garden-based learning"}]
        assert _match(gt, [_search(query="outdoor plant activities")])["exact_match"]
        assert not _match(gt, [_search(query=None)])["exact_match"]

    def test_wrong_operator_or_value_fails(self):
        gt = [{"collection_name": "Staff", "numeric_property_filters": [
            {"property_name": "boardings", "operator": ">=", "value": 560}]}]
        ok = R.IntegerPropertyFilter(property_name="boardings", operator=">=", value=560.0)
        wrong_op = R.IntegerPropertyFilter(property_name="boardings", operator=">", value=560)
        wrong_value = R.IntegerPropertyFilter(property_name="boardings", operator=">=", value=561)
        assert _match(gt, [_search(filters=ok)])["exact_match"]
        assert not _match(gt, [_search(filters=wrong_op)])["exact_match"]
        assert not _match(gt, [_search(filters=wrong_value)])["exact_match"]

    @pytest.mark.parametrize("op,value", [
        (">=", R.DateRangeFrom(date_from="2027-01-04T00:00:00Z", inclusive_from=True)),
        (">", R.DateRangeFrom(date_from="2027-01-04T00:00:00Z", inclusive_from=False)),
        ("<=", R.DateRangeTo(date_to="2027-01-04T00:00:00+00:00", inclusive_to=True)),
        ("<", R.DateRangeTo(date_to="2027-01-04", inclusive_to=False)),
        ("=", R.DateExact(exact_timestamp="2027-01-04T00:00:00Z", operator="=")),
    ])
    def test_date_range_forms(self, op, value):
        gt = [{"collection_name": "Staff", "date_property_filters": [
            {"property_name": "startDate", "operator": op, "value": "2027-01-04"}]}]
        search = _search(filters=R.DatePropertyFilter(property_name="startDate", value=value))
        assert _match(gt, [search])["exact_match"]

    def test_date_between_expands_to_two_comparisons(self):
        gt = [{"collection_name": "Staff", "date_property_filters": [
            {"property_name": "d", "operator": ">=", "value": "2025-01-01"},
            {"property_name": "d", "operator": "<", "value": "2026-01-01"},
        ]}]
        between = R.DateRangeBetween(
            date_from="2025-01-01T00:00:00Z", date_to="2026-01-01T00:00:00Z",
            inclusive_from=True, inclusive_to=False,
        )
        search = _search(filters=R.DatePropertyFilter(property_name="d", value=between))
        assert _match(gt, [search])["exact_match"]

    def test_boolean_not_equals_folds(self):
        gt = [{"collection_name": "Staff", "boolean_property_filters": [
            {"property_name": "meals", "operator": "!=", "value": True}]}]
        search = _search(filters=R.BooleanPropertyFilter(property_name="meals", operator="=", value=False))
        assert _match(gt, [search])["exact_match"]

    def test_or_filter_does_not_match_and(self):
        gt = [{"collection_name": "Staff", "text_property_filters": [
            {"property_name": "a", "operator": "=", "value": "x"},
            {"property_name": "b", "operator": "=", "value": "y"}]}]
        search = _search(filters=R.FilterAndOr(combine="OR", filters=[
            R.TextPropertyFilter(property_name="a", operator="=", value="x"),
            R.TextPropertyFilter(property_name="b", operator="=", value="y"),
        ]))
        assert not _match(gt, [search])["exact_match"]

    def test_aggregation_metric_aliases_and_count_property(self):
        gt = [
            {"collection_name": "Staff", "numeric_property_aggregations": [
                {"property_name": "years", "metric": "MAX"}]},
            {"collection_name": "Staff", "date_property_aggregations": [
                {"property_name": "hired", "metric": "EARLIEST"}]},
            {"collection_name": "Staff", "text_property_aggregations": [
                {"property_name": "name", "metric": "COUNT"}]},
        ]
        aggs = [
            _aggregation(R.IntegerPropertyAggregation(property_name="years", metrics="MAXIMUM")),
            _aggregation(R.DatePropertyAggregation(property_name="hired", metrics="MINIMUM")),
            _aggregation(R.IntegerPropertyAggregation(property_name="years", metrics="COUNT")),
        ]
        assert _match(gt, aggregations=aggs)["exact_match"]

    def test_multi_aggregation_sub_query_needs_every_aggregation(self):
        gt = [{
            "collection_name": "Staff",
            "numeric_property_aggregations": [
                {"property_name": "years", "metric": "MEAN"},
                {"property_name": "years", "metric": "SUM"},
            ],
            "groupby_property": "role",
        }]
        mean = _aggregation(R.IntegerPropertyAggregation(property_name="years", metrics="MEAN"), groupby="role")
        total = _aggregation(R.IntegerPropertyAggregation(property_name="years", metrics="SUM"), groupby="role")
        ungrouped = _aggregation(R.IntegerPropertyAggregation(property_name="years", metrics="SUM"))
        assert _match(gt, aggregations=[mean, total])["exact_match"]
        partial = _match(gt, aggregations=[mean, ungrouped])
        assert not partial["exact_match"]
        assert partial["recall"] == 0.5 and partial["precision"] == 0.5

    def test_count_distinct_matches_groupby_count(self):
        gt = [{
            "collection_name": "Staff",
            "date_property_filters": [{"property_name": "start", "operator": ">=", "value": "2027-01-04"}],
            "numeric_property_aggregations": [{"property_name": "amount", "metric": "SUM"}],
            "text_property_aggregations": [{"property_name": "center", "metric": "COUNT_DISTINCT"}],
        }]
        start = R.DatePropertyFilter(
            property_name="start", value=R.DateRangeFrom(date_from="2027-01-04T00:00:00Z", inclusive_from=True))
        total = _aggregation(R.IntegerPropertyAggregation(property_name="amount", metrics="SUM"), filters=start)
        grouped_count = _aggregation(
            R.TextPropertyAggregation(property_name="childName", metrics="COUNT"), filters=start, groupby="center")
        assert _match(gt, aggregations=[total, grouped_count])["exact_match"]

        ungrouped_count = _aggregation(
            R.TextPropertyAggregation(property_name="center", metrics="COUNT"), filters=start)
        wrong_group = _aggregation(
            R.TextPropertyAggregation(property_name="center", metrics="COUNT"), filters=start, groupby="room")
        assert not _match(gt, aggregations=[total, ungrouped_count])["exact_match"]
        assert not _match(gt, aggregations=[total, wrong_group])["exact_match"]

    def test_top_occurrences_limit_is_compared(self):
        gt = [{"collection_name": "Staff", "text_property_aggregations": [
            {"property_name": "role", "metric": "TOP_OCCURRENCES", "top_occurrences_limit": 3}]}]
        agg = lambda limit: _aggregation(  # noqa: E731
            R.TextPropertyAggregation(property_name="role", metrics="TOP_OCCURRENCES", top_occurrences_limit=limit))
        assert _match(gt, aggregations=[agg(3)])["exact_match"]
        assert not _match(gt, aggregations=[agg(5)])["exact_match"]

    def test_missing_and_extra_sub_queries(self):
        gt = [
            {"collection_name": "Staff", "text_property_filters": [
                {"property_name": "name", "operator": "=", "value": "A"}]},
            {"collection_name": "Centers"},
        ]
        found = _search(filters=R.TextPropertyFilter(property_name="name", operator="=", value="A"))
        extra = _search(query="anything", collection="Centers_Default")
        result = _match(gt, [found, extra])
        assert not result["exact_match"]
        assert result["recall"] == 0.5
        assert result["missing"] == ["SELECT * FROM Centers"]
        assert result["extra"] == ["SELECT * FROM Centers SEARCH <query>"]

    def test_wrong_collection_fails(self):
        gt = [{"collection_name": "Staff"}]
        assert not _match(gt, [_search(collection="Centers_Default")])["exact_match"]


# ============================================================================
# Calculator
# ============================================================================


def _result(gt, searches=None, aggregations=None, answer="ok", classes=("text filter",)):
    query = InMemoryAskQuery(
        question="q",
        ground_truth_answer="a",
        ground_truth_queries=gt,
        metadata={"operator_classes": list(classes), "phrasing": "explicit"},
    )
    return AskResult(
        query=query, system_answer=answer, time_taken=1.0,
        searches=searches, aggregations=aggregations,
    )


class TestQueryExactMatchAskCalculator:
    GT = [{"collection_name": "Staff", "text_property_filters": [
        {"property_name": "name", "operator": "=", "value": "A"}]}]

    def test_scores_and_breakdowns(self):
        right = _search(filters=R.TextPropertyFilter(property_name="name", operator="=", value="A"))
        wrong = _search(filters=R.TextPropertyFilter(property_name="name", operator="=", value="B"))
        results = [
            _result(self.GT, searches=[right], aggregations=[]),
            _result(self.GT, searches=[wrong], aggregations=[], classes=("text filter", "search query")),
            _result(self.GT, answer="[ERROR] boom"),
        ]
        metrics = QueryExactMatchAskCalculator(collection_aliases=ALIASES).compute(results)

        assert metrics["metric"] == "query_exact_match"
        assert metrics["query_exact_match_scores"] == [1, 0, None]
        assert metrics["avg_query_exact_match"] == 0.5
        assert metrics["type_accuracy"] == {"search query": 0.0, "text filter": 0.5}
        assert metrics["misaligned_indices"] == [1]
        assert metrics["query_match_details"][2] is None
        assert metrics["query_match_details"][1]["missing_queries"] == ["SELECT * FROM Staff WHERE name = 'A'"]
        assert "avg_alignment_score" not in metrics

    def test_optional_lm_judge_is_merged(self):
        class StubJudge:
            def compute(self, results):
                return {"avg_alignment_score": 1.0, "alignment_score_scores": [1], "metric": "llm_judge"}

        right = _search(filters=R.TextPropertyFilter(property_name="name", operator="=", value="A"))
        metrics = QueryExactMatchAskCalculator(ALIASES, lm_judge=StubJudge()).compute(
            [_result(self.GT, searches=[right], aggregations=[])]
        )
        assert metrics["metric"] == "query_exact_match"
        assert metrics["avg_query_exact_match"] == 1.0
        assert metrics["avg_alignment_score"] == 1.0

    def test_requires_ground_truth_queries(self):
        result = AskResult(
            query=InMemoryAskQuery(question="q", ground_truth_answer="a"),
            system_answer="x", time_taken=0.1,
        )
        with pytest.raises(ValueError, match="ground_truth_queries"):
            QueryExactMatchAskCalculator().compute([result])


# ============================================================================
# Ask adapter: plan extraction and multi-collection setup
# ============================================================================


class TestWeaviateQueryAgentAskPlan:
    def test_build_response_extracts_searches_and_aggregations(self):
        sdk_response = R.AskModeResponse(
            searches=[R.QueryResultWithCollectionNormalized(
                query="garden", collection="Centers_Default",
                filters=R.BooleanPropertyFilter(property_name="meals", operator="=", value=True),
            )],
            aggregations=[R.AggregationResultWithCollectionNormalized(
                groupby_property="meals", collection="Centers_Default", filters=None,
                aggregation=R.BooleanPropertyAggregation(property_name="open", metrics="PERCENTAGE_TRUE"),
            )],
            usage=R.ModelUnitUsage(model_units=1, usage_in_plan=False, remaining_plan_requests=0),
            total_time=1.0, is_partial_answer=False, missing_information=[],
            final_answer="47%", sources=[],
        )
        response = WeaviateQueryAgentAsk._build_response(sdk_response)

        assert response.final_answer == "47%"
        assert response.searches[0].query == "garden"
        assert response.searches[0].filters["property_name"] == "meals"
        assert response.aggregations[0].groupby_property == "meals"
        assert response.aggregations[0].aggregation["metrics"] == "PERCENTAGE_TRUE"

    def test_explicit_collections_skip_dataset_resolution(self):
        agent = WeaviateQueryAgentAsk(collections=["A_Default", "B_Default"])
        assert agent.collections == ["A_Default", "B_Default"]


# ============================================================================
# Schema-driven collection specs
# ============================================================================


SCHEMA = {
    "use_case": "Test",
    "collections": [
        {
            "name": "Routes",
            "description": "Transit routes.",
            "properties": [
                {"name": "routeDescription", "data_type": "searchable_text", "description": "Prose."},
                {"name": "routeName", "data_type": "filterable_text", "description": "Name.",
                 "is_identifier": True},
                {"name": "hours", "data_type": "int_array", "description": "Hours."},
                {"name": "closures", "data_type": "date_array", "description": "Closures."},
            ],
        },
        {
            "name": "Reports",
            "description": "Daily reports.",
            "kind": "transactional",
            "properties": [
                {"name": "code", "data_type": "filterable_text", "description": "Code.", "is_identifier": True},
                {"name": "serviceDate", "data_type": "date", "description": "Date."},
                {"name": "ok", "data_type": "boolean", "description": "Passed."},
            ],
        },
    ],
}


class TestQueryAgentGymSpecs:
    @pytest.fixture(autouse=True)
    def _schema(self, monkeypatch):
        monkeypatch.setattr(qag_db, "load_query_agent_gym_schema", lambda env: SCHEMA)

    def test_specs_follow_schema(self):
        routes, reports = qag_db.build_query_agent_gym_specs("query-agent-gym/public-transit")

        assert routes.name_fn("query-agent-gym/public-transit") == "Routes"
        assert routes.description == "Transit routes."
        assert routes.uuid_field == "uuid"
        props = {p.name: p for p in routes.properties}
        assert props["routeDescription"].dataType == wvcc.DataType.TEXT
        assert props["routeDescription"].tokenization == wvcc.Tokenization.WORD
        assert props["routeName"].tokenization == wvcc.Tokenization.FIELD
        assert props["routeName"].skip_vectorization
        assert props["routeName"].description == "Name."
        assert props["hours"].dataType == wvcc.DataType.INT_ARRAY
        assert props["closures"].dataType == wvcc.DataType.DATE_ARRAY
        [vector] = routes.vector_config
        assert vector.name == "default"
        assert vector.properties == ["routeDescription"]

        assert {p.dataType for p in reports.properties} >= {wvcc.DataType.DATE, wvcc.DataType.BOOL}
        assert reports.vector_config.vectorizer.vectorizer == wvcc.Vectorizers.NONE

    def test_mapper_converts_dates_and_drops_uuid(self):
        routes, reports = qag_db.build_query_agent_gym_specs("query-agent-gym/public-transit")
        props = reports.item_to_props(
            {"uuid": "u1", "code": "RID-1", "serviceDate": "2025-11-16", "ok": True}
        )
        assert props == {"code": "RID-1", "serviceDate": "2025-11-16T00:00:00Z", "ok": True}
        props = routes.item_to_props({
            "uuid": "u2", "routeDescription": "d", "routeName": "n",
            "hours": [5], "closures": ["2026-12-25"],
        })
        assert props["closures"] == ["2026-12-25T00:00:00Z"]

    def test_collection_names(self):
        assert qag_db.query_agent_gym_collection_names("query-agent-gym/public-transit", tag="Default") == {
            "Routes_Default": "Routes", "Reports_Default": "Reports",
        }

    def test_unknown_environment_raises(self):
        with pytest.raises(ValueError, match="Unknown QueryAgentGym environment"):
            qag_db.build_query_agent_gym_specs("query-agent-gym/nope")

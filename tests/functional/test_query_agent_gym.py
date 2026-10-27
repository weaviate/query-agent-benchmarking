"""Functional tests for QueryAgentGym against the real HuggingFace data.

Downloads ``weaviate/QueryAgentGym`` (~12 MB) and, for every environment,
verifies the schema-driven specs map every object, and that every ground
truth canonicalizes and matches itself under query exact match.

Run with:
    uv run pytest tests/functional/test_query_agent_gym.py -v
"""

import pytest

from query_agent_benchmarking.internal.adapters.database.query_agent_gym import build_query_agent_gym_specs
from query_agent_benchmarking.internal.adapters.dataset.query_agent_gym_loader import (
    QUERY_AGENT_GYM_ENVIRONMENTS,
    load_ask_query_agent_gym,
    load_query_agent_gym_objects,
)
from query_agent_benchmarking.internal.adapters.metrics.query_exact_match import (
    canonicalize_ground_truth,
    score_query_units,
)


DATASETS = [f"query-agent-gym/{env}" for env in QUERY_AGENT_GYM_ENVIRONMENTS]


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_specs_map_every_object(dataset_name):
    environment = dataset_name.split("/", 1)[1]
    objects = load_query_agent_gym_objects(environment)
    specs = build_query_agent_gym_specs(dataset_name)

    assert {s.name_fn(dataset_name) for s in specs} == set(objects)
    for spec in specs:
        rows = objects[spec.name_fn(dataset_name)]
        assert len(rows) == 1000
        declared = {p.name for p in spec.properties}
        for row in rows:
            props = spec.item_to_props(row)
            assert set(props) == declared
            assert row[spec.uuid_field]


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_queries_load_and_self_match(dataset_name):
    queries = load_ask_query_agent_gym(dataset_name)

    assert len(queries) == 50
    for q in queries:
        assert q.question and q.ground_truth_answer and q.query_id
        assert q.metadata["operator_classes"]
        units = canonicalize_ground_truth(q.ground_truth_queries)
        assert units
        assert score_query_units(units, units)["exact_match"]

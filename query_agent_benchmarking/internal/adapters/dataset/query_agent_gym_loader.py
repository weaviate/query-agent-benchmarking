"""Dataset loader for QueryAgentGym (``weaviate/QueryAgentGym`` on HuggingFace).

Unlike the other benchmarks, QueryAgentGym ships its schemas as data: each
environment folder holds a ``schema.json`` alongside one parquet file per
collection and one parquet file per query set. A dataset name picks a query
set and the environments it covers:

- ``query-agent-gym/ask-mode``: the ask mode set (``queries.parquet``), 250 queries
- ``query-agent-gym/search-mode``: the search mode set
  (``search_mode_queries.parquet``), 150 retrieval-only queries
- ``query-agent-gym/edge-case-1``: the exact-value filtering set
  (``edge_case_1.parquet``), 130 cases scored by checks, not an exact plan

over all five environments, or one environment as a third part, e.g.
``query-agent-gym/ask-mode/surf-schools``. Every query set runs against the
same collections, which ``populate_db`` loads as ``query-agent-gym`` (or one
environment, e.g. ``query-agent-gym/surf-schools``).
"""

import json
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

from query_agent_benchmarking.internal.core.domain.models import InMemoryAskQuery, InMemoryQuery


QUERY_AGENT_GYM_REPO = "weaviate/QueryAgentGym"
QUERY_AGENT_GYM = "query-agent-gym"
QUERY_AGENT_GYM_ASK_MODE = f"{QUERY_AGENT_GYM}/ask-mode"
QUERY_AGENT_GYM_SEARCH_MODE = f"{QUERY_AGENT_GYM}/search-mode"
QUERY_AGENT_GYM_EDGE_CASE_1 = f"{QUERY_AGENT_GYM}/edge-case-1"
# Query set name -> the file holding each environment's queries.
QUERY_AGENT_GYM_QUERY_FILES = {
    QUERY_AGENT_GYM_ASK_MODE: "queries.parquet",
    QUERY_AGENT_GYM_SEARCH_MODE: "search_mode_queries.parquet",
    QUERY_AGENT_GYM_EDGE_CASE_1: "edge_case_1.parquet",
}
QUERY_AGENT_GYM_ENVIRONMENTS = (
    "public-transit",
    "tattoo-studios",
    "nonprofits",
    "surf-schools",
    "childcare",
)

# Retrievals can match hundreds of records; the rendered ground-truth answer
# (used only by the optional LM judge) lists at most this many identifiers.
_MAX_LISTED_RECORDS = 50


def _split(dataset_name: str) -> tuple[Optional[str], Optional[str]]:
    """Split a QueryAgentGym name into ``(query set, environment)``; either may be None.

    ``query-agent-gym/ask-mode/surf-schools`` -> ``("query-agent-gym/ask-mode", "surf-schools")``;
    ``query-agent-gym/surf-schools`` (a ``populate_db`` name) -> ``(None, "surf-schools")``.
    """
    if not is_query_agent_gym(dataset_name):
        raise ValueError(f"Not a QueryAgentGym dataset name: {dataset_name!r}")
    rest = dataset_name.split("/")[1:]
    set_name = None
    if rest and f"{QUERY_AGENT_GYM}/{rest[0]}" in QUERY_AGENT_GYM_QUERY_FILES:
        set_name = f"{QUERY_AGENT_GYM}/{rest.pop(0)}"
    if len(rest) > 1:
        raise ValueError(f"Malformed QueryAgentGym dataset name: {dataset_name!r}")
    environment = rest[0] if rest else None
    if environment is not None and environment not in QUERY_AGENT_GYM_ENVIRONMENTS:
        raise ValueError(
            f"Unknown QueryAgentGym query set or environment: {environment!r} in {dataset_name!r}. "
            f"Query sets: {sorted(QUERY_AGENT_GYM_QUERY_FILES)}; environments: {QUERY_AGENT_GYM_ENVIRONMENTS}"
        )
    return set_name, environment


def query_set(dataset_name: str) -> str:
    """The query set a QueryAgentGym dataset name runs, e.g. ``query-agent-gym/ask-mode``."""
    name, _ = _split(dataset_name)
    if name is None:
        raise ValueError(
            f"{dataset_name!r} names no QueryAgentGym query set. "
            f"Use one of {sorted(QUERY_AGENT_GYM_QUERY_FILES)}, optionally with '/<environment>'."
        )
    return name


def is_query_agent_gym(dataset_name: str | None) -> bool:
    return bool(dataset_name) and (
        dataset_name == QUERY_AGENT_GYM or dataset_name.startswith(f"{QUERY_AGENT_GYM}/")
    )


def _is_query_set(dataset_name: str | None, name: str) -> bool:
    return bool(dataset_name) and (dataset_name == name or dataset_name.startswith(f"{name}/"))


def is_query_agent_gym_search_mode(dataset_name: str | None) -> bool:
    return _is_query_set(dataset_name, QUERY_AGENT_GYM_SEARCH_MODE)


def is_query_agent_gym_edge_case(dataset_name: str | None) -> bool:
    return _is_query_set(dataset_name, QUERY_AGENT_GYM_EDGE_CASE_1)


def parse_environments(dataset_name: str) -> tuple[str, ...]:
    """Resolve a QueryAgentGym dataset name to the environments it covers.

    A name without an environment (e.g. ``query-agent-gym/ask-mode`` or
    ``query-agent-gym``) covers every environment; one ending in an environment
    (``query-agent-gym/ask-mode/surf-schools``) covers just that one.
    """
    _, environment = _split(dataset_name)
    return QUERY_AGENT_GYM_ENVIRONMENTS if environment is None else (environment,)


def _download(path: str) -> str:
    return hf_hub_download(QUERY_AGENT_GYM_REPO, path, repo_type="dataset")


@lru_cache(maxsize=None)
def load_query_agent_gym_schema(environment: str) -> dict[str, Any]:
    """Load an environment's ``schema.json`` exactly as generated."""
    with open(_download(f"{environment}/schema.json")) as f:
        return json.load(f)


def _to_python(value: Any) -> Any:
    """Convert numpy scalars/arrays from parquet into plain Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_query_agent_gym_objects(environment: str) -> dict[str, list[dict[str, Any]]]:
    """Load every collection's objects, keyed by collection name.

    Each object is a dict with a ``uuid`` key plus one key per schema property.
    """
    schema = load_query_agent_gym_schema(environment)
    objects: dict[str, list[dict[str, Any]]] = {}
    for collection in schema["collections"]:
        name = collection["name"]
        df = pd.read_parquet(_download(f"{environment}/objects/{name}.parquet"))
        objects[name] = [
            {k: _to_python(v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
    return objects


def query_agent_gym_array_properties(dataset_name: str) -> dict[str, frozenset[str]]:
    """Map each collection name to the names of its array properties."""
    return {
        collection["name"]: frozenset(
            p["name"] for p in collection["properties"] if p["data_type"].endswith("_array")
        )
        for environment in parse_environments(dataset_name)
        for collection in load_query_agent_gym_schema(environment)["collections"]
    }


def query_agent_gym_date_ranges(dataset_name: str) -> dict[str, dict[str, tuple[str, str]]]:
    """Map each collection name to the ``(min, max)`` day of each scalar date property."""
    ranges: dict[str, dict[str, tuple[str, str]]] = {}
    for environment in parse_environments(dataset_name):
        objects = load_query_agent_gym_objects(environment)
        for collection in load_query_agent_gym_schema(environment)["collections"]:
            name = collection["name"]
            ranges[name] = {}
            for prop in collection["properties"]:
                if prop["data_type"] == "date":
                    days = [str(obj[prop["name"]])[:10] for obj in objects[name]]
                    ranges[name][prop["name"]] = (min(days), max(days))
    return ranges


def identifier_property(collection: dict[str, Any]) -> str:
    """Return the name of a schema collection's single identifier property."""
    for prop in collection["properties"]:
        if prop.get("is_identifier"):
            return prop["name"]
    raise ValueError(f"Collection {collection['name']!r} has no identifier property")


def _render_result(
    sql: str,
    result: Any,
    identifiers_by_uuid: dict[str, str],
) -> str:
    """Render one sub-query's reference result as text for the LM judge."""
    if isinstance(result, list):
        names = [identifiers_by_uuid.get(str(u), str(u)) for u in result]
        listed = ", ".join(names[:_MAX_LISTED_RECORDS])
        more = len(names) - _MAX_LISTED_RECORDS
        suffix = f", ... and {more} more" if more > 0 else ""
        return f"{sql}\n-> {len(names)} matching records: {listed}{suffix}"
    return f"{sql}\n-> {json.dumps(result)}"


def _query_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "environment": row["environment"],
        "phrasing": row["phrasing"],
        "plan": row["plan"],
        "num_sub_queries": int(row["num_sub_queries"]),
        "operator_classes": list(row["operator_classes"]),
        "ground_truth_sql": list(row["ground_truth_sql"]),
    }
    if "retrieval_shape" in row:
        metadata["retrieval_shape"] = row["retrieval_shape"]
    return metadata


def load_search_query_agent_gym(dataset_name: str) -> list[InMemoryQuery]:
    """Load search mode queries as ``InMemoryQuery`` objects.

    Each query has a single retrieval sub-query. ``dataset_ids`` holds the UUIDs
    of the records it should return: the sub-query's reference results, which
    for a search are its relevance-labeled records and for a filter every
    matching record. ``ground_truth_queries`` holds the exact ``query_database``
    arguments, for scoring the agent's plan.
    """
    if not is_query_agent_gym_search_mode(dataset_name):
        raise ValueError(f"Not a QueryAgentGym search mode dataset name: {dataset_name!r}")
    queries = []
    for environment in parse_environments(dataset_name):
        df = pd.read_parquet(_download(f"{environment}/{QUERY_AGENT_GYM_QUERY_FILES[QUERY_AGENT_GYM_SEARCH_MODE]}"))
        for row in df.to_dict(orient="records"):
            [relevant_ids] = json.loads(row["reference_results"])
            queries.append(InMemoryQuery(
                question=row["command"],
                dataset_ids=[str(u) for u in relevant_ids],
                query_id=row["query_id"],
                ground_truth_queries=json.loads(row["ground_truth"]),
                metadata=_query_metadata(row),
            ))
    return queries


def _plain(value: Any) -> Any:
    """Convert a parquet struct value (numpy arrays inside) to plain Python."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_edge_case_query_agent_gym(dataset_name: str, include_retry: bool = False) -> list[InMemoryAskQuery]:
    """Load exact-value filtering cases as ``InMemoryAskQuery`` objects.

    Each case is scored by checks rather than an exact ground truth: its
    behavior, target collection, value under test and ``expect`` checks are in
    ``metadata`` (``expect.other_filters`` parsed from JSON). ``ground_truth_answer``
    is a readable summary of the checks.

    Retry cases (``node == "retry"``, B6 and B7) inject a seeded first attempt
    before the agent's zero-results retry, which an agent's ``ask`` call cannot
    take, so they are left out unless ``include_retry`` is set.
    """
    if not is_query_agent_gym_edge_case(dataset_name):
        raise ValueError(f"Not a QueryAgentGym edge case dataset name: {dataset_name!r}")
    queries = []
    for environment in parse_environments(dataset_name):
        df = pd.read_parquet(_download(f"{environment}/{QUERY_AGENT_GYM_QUERY_FILES[QUERY_AGENT_GYM_EDGE_CASE_1]}"))
        for row in df.to_dict(orient="records"):
            if row["node"] == "retry" and not include_retry:
                continue
            expect = _plain(row["expect"])
            expect["other_filters"] = json.loads(expect.get("other_filters") or "[]")
            queries.append(InMemoryAskQuery(
                question=row["command"],
                ground_truth_answer=f"{row['behavior']}: {json.dumps(expect)}",
                query_id=row["query_id"],
                metadata={
                    "environment": row["environment"],
                    "behavior": row["behavior"],
                    "node": row["node"],
                    "collection": row["collection"],
                    "phrasing": row["phrasing"],
                    "value": row["value"],
                    "expect": expect,
                    "num_relevant": len(row["relevant_ids"]),
                    "source": row["source"],
                },
            ))
    return queries


def load_ask_query_agent_gym(dataset_name: str) -> list[InMemoryAskQuery]:
    """Load the queries of every environment in ``dataset_name`` as ``InMemoryAskQuery`` objects.

    ``ground_truth_queries`` holds the exact ``query_database`` arguments used by
    the query exact-match metric. ``ground_truth_answer`` is a text rendering of
    the reference results, used only when the optional LM judge is enabled.
    Each query's environment is in ``metadata["environment"]``.
    """
    query_file = QUERY_AGENT_GYM_QUERY_FILES[query_set(dataset_name)]
    queries = []
    for environment in parse_environments(dataset_name):
        queries.extend(_load_environment_queries(environment, query_file))
    return queries


def _load_environment_queries(environment: str, query_file: str) -> list[InMemoryAskQuery]:
    schema = load_query_agent_gym_schema(environment)
    objects = load_query_agent_gym_objects(environment)

    identifiers_by_uuid: dict[str, str] = {}
    for collection in schema["collections"]:
        id_prop = identifier_property(collection)
        for obj in objects[collection["name"]]:
            identifiers_by_uuid[obj["uuid"]] = str(obj[id_prop])

    df = pd.read_parquet(_download(f"{environment}/{query_file}"))
    queries = []
    for row in df.to_dict(orient="records"):
        sqls = list(row["ground_truth_sql"])
        results = json.loads(row["reference_results"])
        answer = "\n\n".join(
            _render_result(sql, result, identifiers_by_uuid)
            for sql, result in zip(sqls, results)
        )
        queries.append(
            InMemoryAskQuery(
                question=row["command"],
                ground_truth_answer=answer,
                query_id=row["query_id"],
                ground_truth_queries=json.loads(row["ground_truth"]),
                metadata=_query_metadata(row),
            )
        )
    return queries

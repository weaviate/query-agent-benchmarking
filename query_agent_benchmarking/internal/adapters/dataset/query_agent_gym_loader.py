"""Dataset loader for QueryAgentGym (``weaviate/QueryAgentGym`` on HuggingFace).

Unlike the other benchmarks, QueryAgentGym ships its schemas as data: each
environment folder holds a ``schema.json`` alongside one parquet file per
collection and a ``queries.parquet``. Dataset names take the form
``query-agent-gym/<environment>``, e.g. ``query-agent-gym/surf-schools``.
"""

import json
from functools import lru_cache
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from query_agent_benchmarking.internal.core.domain.models import InMemoryAskQuery


QUERY_AGENT_GYM_REPO = "weaviate/QueryAgentGym"
QUERY_AGENT_GYM_PREFIX = "query-agent-gym/"
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


def is_query_agent_gym(dataset_name: str | None) -> bool:
    return bool(dataset_name) and dataset_name.startswith(QUERY_AGENT_GYM_PREFIX)


def parse_environment(dataset_name: str) -> str:
    """Extract and validate the environment from ``query-agent-gym/<environment>``."""
    if not is_query_agent_gym(dataset_name):
        raise ValueError(f"Not a QueryAgentGym dataset name: {dataset_name!r}")
    environment = dataset_name[len(QUERY_AGENT_GYM_PREFIX):]
    if environment not in QUERY_AGENT_GYM_ENVIRONMENTS:
        raise ValueError(
            f"Unknown QueryAgentGym environment: {environment!r}. "
            f"Supported: {QUERY_AGENT_GYM_ENVIRONMENTS}"
        )
    return environment


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


def load_ask_query_agent_gym(dataset_name: str) -> list[InMemoryAskQuery]:
    """Load an environment's queries as ``InMemoryAskQuery`` objects.

    ``ground_truth_queries`` holds the exact ``query_database`` arguments used by
    the query exact-match metric. ``ground_truth_answer`` is a text rendering of
    the reference results, used only when the optional LM judge is enabled.
    """
    environment = parse_environment(dataset_name)
    schema = load_query_agent_gym_schema(environment)
    objects = load_query_agent_gym_objects(environment)

    identifiers_by_uuid: dict[str, str] = {}
    for collection in schema["collections"]:
        id_prop = identifier_property(collection)
        for obj in objects[collection["name"]]:
            identifiers_by_uuid[obj["uuid"]] = str(obj[id_prop])

    df = pd.read_parquet(_download(f"{environment}/queries.parquet"))
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
                metadata={
                    "environment": row["environment"],
                    "phrasing": row["phrasing"],
                    "plan": row["plan"],
                    "num_sub_queries": int(row["num_sub_queries"]),
                    "operator_classes": list(row["operator_classes"]),
                    "ground_truth_sql": sqls,
                },
            )
        )
    return queries

"""Query exact-match metric for QueryAgentGym.

Scores the database queries an agent *issued* (the Query Agent's ``searches``
and ``aggregations``) against the exact ground-truth queries, rather than
scoring the final natural-language answer.

Both sides are canonicalized into a set of hashable "query units":

- a retrieval: ``("search", collection, has_search_query, filters)``
- one aggregation: ``("aggregate", collection, groupby, filters, (property, metric, limit))``

Ground-truth sub-queries with several aggregations expand into one unit per
aggregation, matching the SDK's normalized one-aggregation-per-entry form.
``filters`` is a frozenset of leaves combined with AND; OR groups stay nested.

Search query *text* is not compared: the ground truth's wording is one valid
phrasing, not the target. Only whether a search was issued matters. Sorting
is ignored, since it does not change which records match.

Canonicalization only folds forms that are semantically identical:

- operators are upper-cased (``contains_any`` -> ``CONTAINS_ANY``)
- ``CONTAINS_ALL`` / ``=`` with a single array value -> ``CONTAINS_ANY``
- boolean ``!= v`` -> ``= not v``
- SQL ``LIKE`` wildcards ``%`` -> Weaviate's ``*``
- date ranges (``date_from`` / ``date_to`` / inclusive flags) -> comparison operators,
  and every date -> a UTC ``YYYY-MM-DDTHH:MM:SSZ`` timestamp
- numbers -> float; ``MINIMUM``/``MAXIMUM``/``EARLIEST``/``LATEST`` -> ``MIN``/``MAX``
- ``COUNT`` ignores its property (no QueryAgentGym value is null, so any property counts all records)
- ``COUNT_DISTINCT(p)`` -> ``COUNT`` grouped by ``p``, the form the Query Agent issues
  (Weaviate has no distinct-count metric; the number of groups is the answer)
"""

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from query_agent_benchmarking.internal.core.domain.models import AgentAggregation, AgentSearch


Unit = tuple
Filters = frozenset

_ARRAY_FILTER_KEYS = (
    "array_property_filters",
    "numeric_array_property_filters",
    "boolean_array_property_filters",
    "date_array_property_filters",
)
_DATE_FILTER_KEYS = ("date_property_filters", "date_array_property_filters")
_METRIC_ALIASES = {
    "MINIMUM": "MIN",
    "MAXIMUM": "MAX",
    "EARLIEST": "MIN",
    "LATEST": "MAX",
}


# ============================================================================
# Value / leaf canonicalization
# ============================================================================

def _canonical_date(value: Any) -> str:
    """Normalize an ISO date or timestamp to a UTC ``YYYY-MM-DDTHH:MM:SSZ`` string."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_scalar(value: Any, is_date: bool) -> Any:
    if is_date:
        return _canonical_date(value)
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _leaf(prop: str, op: str, value: Any, is_date: bool = False, is_array: bool = False) -> tuple:
    """Build a canonical filter leaf ``(property, operator, value)``."""
    op = str(op).upper()
    if is_array or isinstance(value, (list, tuple)):
        values = value if isinstance(value, (list, tuple)) else [value]
        canonical = frozenset(_canonical_scalar(v, is_date) for v in values)
        if len(canonical) == 1 and op in ("CONTAINS_ALL", "="):
            op = "CONTAINS_ANY"
        return (prop, op, canonical)

    canonical = _canonical_scalar(value, is_date)
    if isinstance(canonical, bool) and op == "!=":
        return (prop, "=", not canonical)
    if op == "LIKE" and isinstance(canonical, str):
        canonical = canonical.replace("%", "*")
    return (prop, op, canonical)


# ============================================================================
# Ground truth (query_database arguments)
# ============================================================================

def _ground_truth_filters(sub_query: dict[str, Any]) -> Filters:
    leaves = []
    for key, items in sub_query.items():
        if not key.endswith("_filters"):
            continue
        is_date = key in _DATE_FILTER_KEYS
        is_array = key in _ARRAY_FILTER_KEYS
        for f in items:
            value = f["values"] if "values" in f else f["value"]
            leaves.append(_leaf(f["property_name"], f["operator"], value, is_date, is_array))
    return frozenset(leaves)


def _aggregation_unit(
    collection: str,
    groupby: Optional[str],
    filters: Filters,
    prop: str,
    metric: str,
    limit: Optional[int],
) -> Unit:
    metric = _METRIC_ALIASES.get(str(metric).upper(), str(metric).upper())
    if metric == "COUNT_DISTINCT" and groupby is None:
        groupby, metric = prop, "COUNT"
    if metric == "COUNT":
        prop = "*"
    if metric != "TOP_OCCURRENCES":
        limit = None
    return ("aggregate", collection, groupby, filters, (prop, metric, limit))


def canonicalize_ground_truth(sub_queries: list[dict[str, Any]]) -> list[Unit]:
    """Canonicalize a list of ground-truth ``query_database`` argument objects."""
    units: list[Unit] = []
    for sq in sub_queries:
        collection = sq["collection_name"]
        filters = _ground_truth_filters(sq)
        aggregations = [
            a for key, items in sq.items() if key.endswith("_aggregations") for a in items
        ]
        if not aggregations:
            units.append(("search", collection, bool(sq.get("search_query")), filters))
            continue
        groupby = sq.get("groupby_property")
        for a in aggregations:
            units.append(_aggregation_unit(
                collection, groupby, filters,
                a["property_name"], a["metric"], a.get("top_occurrences_limit"),
            ))
    return units


# ============================================================================
# Predicted (Weaviate Query Agent normalized searches / aggregations)
# ============================================================================

def _predicted_date_leaves(prop: str, value: dict[str, Any]) -> list[tuple]:
    """Convert an SDK ``DatePropertyFilter`` value into comparison leaves."""
    if "exact_timestamp" in value:
        return [_leaf(prop, value["operator"], value["exact_timestamp"], is_date=True)]
    leaves = []
    if "date_from" in value:
        op = ">=" if value.get("inclusive_from") else ">"
        leaves.append(_leaf(prop, op, value["date_from"], is_date=True))
    if "date_to" in value:
        op = "<=" if value.get("inclusive_to") else "<"
        leaves.append(_leaf(prop, op, value["date_to"], is_date=True))
    return leaves


def _predicted_filter_nodes(node: Any) -> list[Any]:
    """Convert an SDK filter tree into nodes to be AND-ed together.

    AND groups are flattened into their children; OR groups become a single
    ``("OR", frozenset(children))`` node.
    """
    if node is None:
        return []
    if "combine" in node:
        children: list[Any] = []
        if node["combine"] == "AND":
            for child in node["filters"]:
                children.extend(_predicted_filter_nodes(child))
            return children
        branches = []
        for child in node["filters"]:
            nodes = _predicted_filter_nodes(child)
            branches.append(nodes[0] if len(nodes) == 1 else ("AND", frozenset(nodes)))
        if len(branches) == 1:
            return [branches[0]]
        return [("OR", frozenset(branches))]

    prop = node.get("property_name", "")
    filter_type = node.get("filter_type") or ""
    if filter_type == "date_range":
        return _predicted_date_leaves(prop, node["value"])
    if filter_type == "is_null":
        return [(prop, "IS_NULL", bool(node.get("is_null")))]
    if "operator" not in node:
        # Geo and unknown filter types: keep them verbatim so they never match by accident.
        return [(prop, filter_type.upper(), repr(sorted(node.items())))]
    return [_leaf(
        prop,
        node["operator"],
        node.get("value"),
        is_date=filter_type == "date_array",
        is_array=filter_type.endswith("_array"),
    )]


def _predicted_filters(filters: Any) -> Filters:
    return frozenset(_predicted_filter_nodes(filters))


def canonicalize_predicted(
    searches: Optional[Iterable[AgentSearch]],
    aggregations: Optional[Iterable[AgentAggregation]],
    collection_aliases: Optional[dict[str, str]] = None,
) -> list[Unit]:
    """Canonicalize the searches and aggregations an agent issued.

    Args:
        searches: The agent's normalized searches.
        aggregations: The agent's normalized aggregations.
        collection_aliases: Maps Weaviate collection names (e.g. ``SurfBeaches_Default``)
            back to the dataset's collection names (``SurfBeaches``).
    """
    aliases = collection_aliases or {}
    units: list[Unit] = []
    for s in searches or []:
        filters = set(_predicted_filters(s.filters))
        if s.uuid_value:
            filters.add(("_id", "=", s.uuid_value))
        collection = aliases.get(s.collection, s.collection)
        units.append(("search", collection, bool(s.query), frozenset(filters)))
    for a in aggregations or []:
        agg = a.aggregation or {}
        units.append(_aggregation_unit(
            aliases.get(a.collection, a.collection),
            a.groupby_property,
            _predicted_filters(a.filters),
            agg.get("property_name", ""),
            agg.get("metrics", ""),
            agg.get("top_occurrences_limit"),
        ))
    return units


# ============================================================================
# Scoring and rendering
# ============================================================================

def score_query_units(ground_truth: list[Unit], predicted: list[Unit]) -> dict[str, Any]:
    """Compare canonical unit sets.

    ``exact_match`` requires the predicted set to equal the ground-truth set
    (every sub-query issued, nothing extra). ``recall`` / ``precision`` give
    partial credit at the unit level.
    """
    gt, pred = set(ground_truth), set(predicted)
    matched = gt & pred
    return {
        "exact_match": gt == pred,
        "recall": len(matched) / len(gt) if gt else 1.0,
        "precision": len(matched) / len(pred) if pred else 0.0,
        "missing": [render_unit(u) for u in sorted(gt - pred, key=repr)],
        "extra": [render_unit(u) for u in sorted(pred - gt, key=repr)],
    }


def _render_value(value: Any) -> str:
    if isinstance(value, frozenset):
        return "(" + ", ".join(sorted(_render_value(v) for v in value)) + ")"
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).upper() if isinstance(value, bool) else str(value)


def _render_node(node: tuple) -> str:
    if node[0] in ("AND", "OR") and isinstance(node[1], frozenset):
        parts = sorted(_render_node(child) for child in node[1])
        return "(" + f" {node[0]} ".join(parts) + ")"
    prop, op, value = node
    return f"{prop} {op} {_render_value(value)}"


def _render_where(filters: Filters) -> str:
    if not filters:
        return ""
    return " WHERE " + " AND ".join(sorted(_render_node(n) for n in filters))


def render_unit(unit: Unit) -> str:
    """Render a canonical unit as a readable SQL-like string."""
    if unit[0] == "search":
        _, collection, has_search, filters = unit
        search = " SEARCH <query>" if has_search else ""
        return f"SELECT * FROM {collection}{search}{_render_where(filters)}"
    _, collection, groupby, filters, (prop, metric, limit) = unit
    args = prop if limit is None else f"{prop}, limit={limit}"
    group = f" GROUP BY {groupby}" if groupby else ""
    return f"SELECT {metric}({args}) FROM {collection}{_render_where(filters)}{group}"

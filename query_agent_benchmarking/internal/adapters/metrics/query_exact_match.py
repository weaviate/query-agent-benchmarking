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

Canonicalization only folds forms that match the same records:

- operators are upper-cased (``contains_any`` -> ``CONTAINS_ANY``)
- ``CONTAINS_ALL`` / ``=`` with a single array value -> ``CONTAINS_ANY``. This
  includes a scalar-typed ``=`` filter on an array property (given
  ``array_properties``): Weaviate's ``Equal`` on an array matches any element.
- boolean ``!= v`` -> ``= not v``
- SQL ``LIKE`` wildcards ``%`` -> Weaviate's ``*``
- date ranges (``date_from`` / ``date_to`` / inclusive flags) -> comparison operators.
  QueryAgentGym dates are calendar days, so every date comparison becomes an
  inclusive day bound: ``< 2025-04-10``, ``<= 2025-04-09T23:59:59`` and
  ``<= 2025-04-09`` all become ``<= 2025-04-09``. Array dates -> UTC timestamps.
- a date bound no record falls outside of (given ``date_ranges``) is dropped,
  e.g. ``>= 0001-01-01``, ``<= 9999-12-31``, or ``<= today`` on past-only dates
- numbers -> float; ``MINIMUM``/``MAXIMUM``/``EARLIEST``/``LATEST`` -> ``MIN``/``MAX``
- ``COUNT`` ignores its property (no QueryAgentGym value is null, so any property counts all records)
- ``COUNT_DISTINCT(p)`` -> ``COUNT`` grouped by ``p``, the form the Query Agent issues
  (Weaviate has no distinct-count metric; the number of groups is the answer)
- ``TOTAL_TRUE(p)`` / ``TOTAL_FALSE(p)`` on a scalar boolean -> ``COUNT`` filtered
  on ``p = true`` / ``p = false``. Not on boolean arrays, where it counts values.

``array_properties`` and ``date_ranges`` are keyed by dataset collection name.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Optional

from query_agent_benchmarking.internal.core.domain.models import AgentAggregation, AgentSearch


Unit = tuple
Filters = frozenset
ArrayProperties = dict[str, frozenset[str]]  # collection -> its array property names
DateRanges = dict[str, dict[str, tuple[str, str]]]  # collection -> date property -> (min, max) day

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

def _parse_date(value: Any) -> datetime:
    """Parse an ISO date or timestamp as a UTC datetime."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_date(value: Any) -> str:
    """Normalize an ISO date or timestamp to a UTC ``YYYY-MM-DDTHH:MM:SSZ`` string."""
    return _parse_date(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _shift_day(day: date, days: int) -> date:
    try:
        return day + timedelta(days=days)
    except OverflowError:
        return day  # 0001-01-01 / 9999-12-31: already beyond any record


def _day_bound(op: str, value: Any) -> tuple[str, str]:
    """Turn a comparison on a calendar-day property into an inclusive day bound.

    Returns ``(">=" | "<=", "YYYY-MM-DD")``, or ``("=", day)`` for an exact
    match on midnight. Any other operator keeps its full timestamp.
    """
    ts = _parse_date(value)
    day = ts.date()
    midnight = ts == datetime.combine(day, time(), timezone.utc)
    if op == ">=":
        return ">=", (day if midnight else _shift_day(day, 1)).isoformat()
    if op == ">":
        return ">=", _shift_day(day, 1).isoformat()
    if op == "<=":
        return "<=", day.isoformat()
    if op == "<":
        return "<=", (_shift_day(day, -1) if midnight else day).isoformat()
    if op == "=" and midnight:
        return "=", day.isoformat()
    return op, _canonical_date(value)


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

    if is_date:
        return (prop, *_day_bound(op, value))
    canonical = _canonical_scalar(value, is_date)
    if isinstance(canonical, bool) and op == "!=":
        return (prop, "=", not canonical)
    if op == "LIKE" and isinstance(canonical, str):
        canonical = canonical.replace("%", "*")
    return (prop, op, canonical)


# ============================================================================
# Ground truth (query_database arguments)
# ============================================================================

def _drop_vacuous_date_bounds(filters: Filters, ranges: dict[str, tuple[str, str]]) -> Filters:
    """Drop top-level date bounds that every record satisfies."""
    def vacuous(node: Any) -> bool:
        if len(node) != 3 or node[0] not in ranges:
            return False
        low, high = ranges[node[0]]
        op, day = node[1], node[2]
        return (op == ">=" and day <= low) or (op == "<=" and day >= high)

    return frozenset(n for n in filters if not vacuous(n))


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
    array_props: frozenset[str] = frozenset(),
) -> Unit:
    metric = _METRIC_ALIASES.get(str(metric).upper(), str(metric).upper())
    if metric == "COUNT_DISTINCT" and groupby is None:
        groupby, metric = prop, "COUNT"
    if metric in ("TOTAL_TRUE", "TOTAL_FALSE") and prop not in array_props:
        filters = filters | {(prop, "=", metric == "TOTAL_TRUE")}
        metric = "COUNT"
    if metric == "COUNT":
        prop = "*"
    if metric != "TOP_OCCURRENCES":
        limit = None
    return ("aggregate", collection, groupby, filters, (prop, metric, limit))


def canonicalize_ground_truth(
    sub_queries: list[dict[str, Any]],
    array_properties: Optional[ArrayProperties] = None,
    date_ranges: Optional[DateRanges] = None,
) -> list[Unit]:
    """Canonicalize a list of ground-truth ``query_database`` argument objects."""
    array_properties = array_properties or {}
    date_ranges = date_ranges or {}
    units: list[Unit] = []
    for sq in sub_queries:
        collection = sq["collection_name"]
        filters = _drop_vacuous_date_bounds(_ground_truth_filters(sq), date_ranges.get(collection, {}))
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
                array_properties.get(collection, frozenset()),
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


def _predicted_filter_nodes(node: Any, array_props: frozenset[str] = frozenset()) -> list[Any]:
    """Convert an SDK filter tree into nodes to be AND-ed together.

    AND groups are flattened into their children; OR groups become a single
    ``("OR", frozenset(children))`` node. ``array_props`` names the collection's
    array properties, so a scalar-typed filter on one is treated as an array filter.
    """
    if node is None:
        return []
    if "combine" in node:
        children: list[Any] = []
        if node["combine"] == "AND":
            for child in node["filters"]:
                children.extend(_predicted_filter_nodes(child, array_props))
            return children
        branches = []
        for child in node["filters"]:
            nodes = _predicted_filter_nodes(child, array_props)
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
        is_array=filter_type.endswith("_array") or prop in array_props,
    )]


def _predicted_filters(
    filters: Any,
    array_props: frozenset[str] = frozenset(),
    date_ranges: Optional[dict[str, tuple[str, str]]] = None,
) -> Filters:
    nodes = frozenset(_predicted_filter_nodes(filters, array_props))
    return _drop_vacuous_date_bounds(nodes, date_ranges or {})


def canonicalize_filters(
    filters: Any,
    array_props: frozenset[str] = frozenset(),
    date_ranges: Optional[dict[str, tuple[str, str]]] = None,
) -> Filters:
    """Canonicalize one SDK filter tree into AND-ed nodes (leaves and OR/AND groups)."""
    return _predicted_filters(filters, array_props, date_ranges)


def canonical_leaf(prop: str, op: str, value: Any, is_date: bool = False, is_array: bool = False) -> tuple:
    """Canonicalize one ``(property, operator, value)`` filter, as for issued filters."""
    return _leaf(prop, op, value, is_date, is_array)


def canonicalize_predicted(
    searches: Optional[Iterable[AgentSearch]],
    aggregations: Optional[Iterable[AgentAggregation]],
    collection_aliases: Optional[dict[str, str]] = None,
    array_properties: Optional[ArrayProperties] = None,
    date_ranges: Optional[DateRanges] = None,
) -> list[Unit]:
    """Canonicalize the searches and aggregations an agent issued.

    Args:
        searches: The agent's normalized searches.
        aggregations: The agent's normalized aggregations.
        collection_aliases: Maps Weaviate collection names (e.g. ``SurfBeaches_Default``)
            back to the dataset's collection names (``SurfBeaches``).
        array_properties: Each collection's array property names.
        date_ranges: Each collection's ``(min, max)`` day per date property.
    """
    aliases = collection_aliases or {}
    array_properties = array_properties or {}
    date_ranges = date_ranges or {}
    units: list[Unit] = []
    for s in searches or []:
        collection = aliases.get(s.collection, s.collection)
        array_props = array_properties.get(collection, frozenset())
        filters = set(_predicted_filters(s.filters, array_props, date_ranges.get(collection)))
        if s.uuid_value:
            filters.add(("_id", "=", s.uuid_value))
        units.append(("search", collection, bool(s.query), frozenset(filters)))
    for a in aggregations or []:
        agg = a.aggregation or {}
        collection = aliases.get(a.collection, a.collection)
        array_props = array_properties.get(collection, frozenset())
        units.append(_aggregation_unit(
            collection,
            a.groupby_property,
            _predicted_filters(a.filters, array_props, date_ranges.get(collection)),
            agg.get("property_name", ""),
            agg.get("metrics", ""),
            agg.get("top_occurrences_limit"),
            array_props,
        ))
    return units


# ============================================================================
# Scoring and rendering
# ============================================================================

def score_query_units(ground_truth: list[Unit], predicted: list[Unit]) -> dict[str, Any]:
    """Compare canonical unit sets.

    ``recall_match`` requires every ground-truth sub-query to be issued; extra
    sub-queries are allowed. ``exact_match`` also forbids extras (the predicted
    set equals the ground-truth set). ``recall`` / ``precision`` give partial
    credit at the unit level.
    """
    gt, pred = set(ground_truth), set(predicted)
    matched = gt & pred
    return {
        "recall_match": gt <= pred,
        "exact_match": gt == pred,
        "recall": len(matched) / len(gt) if gt else 1.0,
        "precision": len(matched) / len(pred) if pred else 0.0,
        "missing": [render_unit(u) for u in sorted(gt - pred, key=repr)],
        "extra": [render_unit(u) for u in sorted(pred - gt, key=repr)],
    }


def match_issued_queries(
    ground_truth_queries: list[dict[str, Any]],
    searches: Optional[Iterable[AgentSearch]],
    aggregations: Optional[Iterable[AgentAggregation]],
    collection_aliases: Optional[dict[str, str]] = None,
    array_properties: Optional[ArrayProperties] = None,
    date_ranges: Optional[DateRanges] = None,
) -> dict[str, Any]:
    """Canonicalize and score one query's issued plan against its ground truth.

    Returns :func:`score_query_units`'s result plus ``ground_truth_queries`` and
    ``predicted_queries``, the rendered canonical units on each side.
    """
    ground_truth = canonicalize_ground_truth(ground_truth_queries, array_properties, date_ranges)
    predicted = canonicalize_predicted(
        searches, aggregations, collection_aliases, array_properties, date_ranges,
    )
    return {
        **score_query_units(ground_truth, predicted),
        "ground_truth_queries": [render_unit(u) for u in ground_truth],
        "predicted_queries": [render_unit(u) for u in predicted],
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

"""Check-based scoring for QueryAgentGym's exact-value filtering cases (``edge_case_1``).

Each case gives an exact value and checks one decision: does the agent filter
on it, or put it in the search query? Cases carry ``expect`` checks rather than
an exact plan:

- ``value_filter``: how the value must be filtered, on ``value_properties``:
  ``eq`` (an ``=`` filter), ``like_prefix`` (``LIKE '<value>*'``), ``or_eq``
  (an OR of one ``=`` per property), ``none`` (no ``=`` or ``LIKE`` anywhere
  carries the value), or ``no_eq`` (no ``=`` on the term; a search, or for a
  count at most one single-keyword ``LIKE``)
- ``query_contains``: strings that must appear verbatim in the search query
- ``query_null``: the search query must be empty
- ``other_filters``: filters that must also be AND-ed in

The agent may issue several searches and aggregations. Positive checks
(the value filter, ``query_contains``, ``query_null``, ``other_filters``) must
all hold for **one** issued query on the case's collection; extra queries are
not penalized. Negative checks (``none``, ``no_eq``) must hold for **every**
issued query. Filters are canonicalized as in :mod:`query_exact_match`.
"""

from typing import Any, Iterable, Optional

import numpy as np

from query_agent_benchmarking.internal.adapters.metrics.query_exact_match import (
    canonical_leaf,
    canonicalize_filters,
    canonicalize_predicted,
    render_unit,
)
from query_agent_benchmarking.internal.core.domain.models import AgentAggregation, AgentSearch, AskResult


def _issued_queries(
    searches: Optional[Iterable[AgentSearch]],
    aggregations: Optional[Iterable[AgentAggregation]],
    aliases: dict[str, str],
    array_properties: dict[str, frozenset[str]],
    date_ranges: dict[str, dict[str, tuple[str, str]]],
) -> list[dict[str, Any]]:
    """Each issued search or aggregation as ``{collection, query, filters}``."""
    issued = []
    for s in searches or []:
        collection = aliases.get(s.collection, s.collection)
        issued.append({
            "collection": collection,
            "query": s.query or None,
            "filters": canonicalize_filters(
                s.filters, array_properties.get(collection, frozenset()), date_ranges.get(collection),
            ),
        })
    for a in aggregations or []:
        collection = aliases.get(a.collection, a.collection)
        issued.append({
            "collection": collection,
            "query": None,
            "filters": canonicalize_filters(
                a.filters, array_properties.get(collection, frozenset()), date_ranges.get(collection),
            ),
        })
    return issued


def _leaves(nodes: Iterable[Any]) -> Iterable[tuple]:
    """Every filter leaf, descending into AND/OR groups."""
    for node in nodes:
        if node[0] in ("AND", "OR") and isinstance(node[1], frozenset):
            yield from _leaves(node[1])
        else:
            yield node


def _carries(leaf: tuple, value: str) -> bool:
    """Whether a leaf's value contains ``value`` (case-insensitive, ignoring ``*``)."""
    values = leaf[2] if isinstance(leaf[2], frozenset) else [leaf[2]]
    needle = value.lower()
    return any(isinstance(v, str) and needle in v.replace("*", "").lower() for v in values)


def _value_filter_holds(kind: str, filters: frozenset, value: str, properties: list[str]) -> bool:
    """Positive value-filter check on one issued query's top-level filters."""
    if kind == "eq":
        return any((p, "=", value) in filters for p in properties)
    if kind == "like_prefix":
        return any((p, "LIKE", f"{value}*") in filters for p in properties)
    if kind == "or_eq":
        expected = frozenset((p, "=", value) for p in properties)
        return ("OR", expected) in filters
    return True  # "none" / "no_eq" are checked across every issued query


def check_edge_case(
    metadata: dict[str, Any],
    searches: Optional[Iterable[AgentSearch]],
    aggregations: Optional[Iterable[AgentAggregation]],
    collection_aliases: Optional[dict[str, str]] = None,
    array_properties: Optional[dict[str, frozenset[str]]] = None,
    date_ranges: Optional[dict[str, dict[str, tuple[str, str]]]] = None,
) -> dict[str, Any]:
    """Score one case's issued queries against its ``expect`` checks.

    Returns ``passed``, ``checks`` (each check -> whether it held), and
    ``failures`` (the names of the checks that did not).
    """
    array_properties = array_properties or {}
    date_ranges = date_ranges or {}
    expect = metadata["expect"]
    value = metadata["value"]
    collection = metadata["collection"]
    kind = expect["value_filter"]
    properties = list(expect.get("value_properties") or [])
    query_contains = list(expect.get("query_contains") or [])

    issued = _issued_queries(searches, aggregations, collection_aliases or {}, array_properties, date_ranges)
    on_collection = [q for q in issued if q["collection"] == collection]

    array_props = array_properties.get(collection, frozenset())
    date_props = set(date_ranges.get(collection, {}))
    other_filters = {
        canonical_leaf(
            f["property_name"], f["operator"], f["value"],
            is_date=f["property_name"] in date_props, is_array=f["property_name"] in array_props,
        )
        for f in expect.get("other_filters") or []
    }

    # Positive checks, per issued query on the case's collection.
    positive: dict[str, Any] = {}
    if kind in ("eq", "like_prefix", "or_eq"):
        positive["value_filter"] = lambda q: _value_filter_holds(kind, q["filters"], value, properties)
    if query_contains:
        positive["query_contains"] = lambda q: bool(q["query"]) and all(t in q["query"] for t in query_contains)
    if expect.get("query_null"):
        positive["query_null"] = lambda q: not q["query"]
    if other_filters:
        positive["other_filters"] = lambda q: other_filters <= q["filters"]

    checks: dict[str, bool] = {"collection": bool(on_collection)}
    for name, holds in positive.items():
        checks[name] = any(holds(q) for q in on_collection)
    if positive:
        checks["same_query"] = any(all(holds(q) for holds in positive.values()) for q in on_collection)

    # Negative checks, across every issued query.
    all_leaves = [leaf for q in issued for leaf in _leaves(q["filters"])]
    if kind == "none":
        checks["value_not_filtered"] = not any(
            leaf[1] in ("=", "LIKE") and _carries(leaf, value) for leaf in all_leaves
        )
    elif kind == "no_eq":
        checks["no_eq_on_term"] = not any(leaf[1] == "=" and _carries(leaf, value) for leaf in all_leaves)
        likes = [leaf for leaf in all_leaves if leaf[1] == "LIKE" and _carries(leaf, value)]
        single_keyword_like = len(likes) == 1 and " " not in str(likes[0][2]).replace("*", "").strip()
        searched = any(q["query"] for q in on_collection)
        checks["searched_or_one_like"] = (
            searched or (metadata.get("node") == "aggregate" and single_keyword_like)
        ) and len(likes) <= 1

    failures = [name for name, held in checks.items() if not held]
    return {"passed": not failures, "checks": checks, "failures": failures}


def _group_means(scores: dict[str, list[int]]) -> dict[str, float]:
    return {group: float(np.mean(values)) for group, values in sorted(scores.items())}


class QueryAgentGymEdgeCaseAskCalculator:
    """Ask metrics calculator for QueryAgentGym's exact-value filtering cases.

    A case scores 1 when every check holds (see :func:`check_edge_case`).
    ``avg_edge_case_pass`` is the headline, broken down per behavior (B1 to B9,
    as ``type_accuracy``), environment and phrasing, with how often each check
    failed under ``check_failure_counts``. Failed agent calls are left out.

    Args:
        collection_aliases: Maps Weaviate collection names back to dataset collection names.
        array_properties: Each dataset collection's array property names.
        date_ranges: Each dataset collection's ``(min, max)`` day per date property.
    """

    def __init__(
        self,
        collection_aliases: Optional[dict[str, str]] = None,
        array_properties: Optional[dict[str, frozenset[str]]] = None,
        date_ranges: Optional[dict[str, dict[str, tuple[str, str]]]] = None,
    ):
        self.collection_aliases = collection_aliases or {}
        self.array_properties = array_properties or {}
        self.date_ranges = date_ranges or {}

    def compute(self, results: list[AskResult]) -> dict:
        print(f"\n\033[94mChecking {len(results)} exact-value filtering cases...\033[0m")
        scores: list[Optional[int]] = []
        details: list[Optional[dict]] = []
        query_times = []
        misaligned_indices = []
        by_behavior: dict[str, list[int]] = {}
        by_environment: dict[str, list[int]] = {}
        by_phrasing: dict[str, list[int]] = {}
        failure_counts: dict[str, int] = {}

        for i, result in enumerate(results):
            metadata = result.query.metadata or {}
            if "expect" not in metadata:
                raise ValueError(
                    f"Query {i} has no 'expect' checks; this calculator requires "
                    "QueryAgentGym edge case queries."
                )
            if result.system_answer.startswith("[ERROR]"):
                scores.append(None)
                details.append(None)
                continue

            check = check_edge_case(
                metadata, result.searches, result.aggregations,
                self.collection_aliases, self.array_properties, self.date_ranges,
            )
            score = 1 if check["passed"] else 0
            scores.append(score)
            query_times.append(result.time_taken)
            if not score:
                misaligned_indices.append(i)
            by_behavior.setdefault(metadata["behavior"], []).append(score)
            by_environment.setdefault(metadata["environment"], []).append(score)
            by_phrasing.setdefault(metadata["phrasing"], []).append(score)
            for name in check["failures"]:
                failure_counts[name] = failure_counts.get(name, 0) + 1

            issued = canonicalize_predicted(
                result.searches, result.aggregations, self.collection_aliases,
                self.array_properties, self.date_ranges,
            )
            details.append({
                "edge_case_pass": score,
                "checks": check["checks"],
                "failed_checks": check["failures"],
                "issued_queries": [render_unit(u) for u in issued],
            })

        valid = [s for s in scores if s is not None]
        results_dict = {
            "avg_query_time": float(np.mean(query_times)) if query_times else 0,
            "query_times": query_times,
            "misaligned_indices": misaligned_indices,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "metric": "edge_case_pass",
            "avg_edge_case_pass": float(np.mean(valid)) if valid else 0,
            "edge_case_pass_scores": scores,
            "type_accuracy": _group_means(by_behavior),
            "type_counts": {b: len(v) for b, v in sorted(by_behavior.items())},
            "environment_accuracy": _group_means(by_environment),
            "phrasing_accuracy": _group_means(by_phrasing),
            "check_failure_counts": dict(sorted(failure_counts.items())),
            "query_match_details": details,
        }
        self._print_summary(results_dict, len(valid))
        return results_dict

    @staticmethod
    def _print_summary(results_dict: dict, num_evaluated: int) -> None:
        print("\n\033[92m===== Exact-Value Filtering (edge_case_1) Results =====\033[0m")
        print(f"Number of cases evaluated: {num_evaluated}")
        print(f"Edge Case Pass Rate: {results_dict['avg_edge_case_pass']:.2%}")
        print("\n\033[93mPer Behavior:\033[0m")
        for behavior, acc in results_dict["type_accuracy"].items():
            print(f"  {behavior}: {acc:.2%} ({results_dict['type_counts'][behavior]} cases)")
        print("\n\033[93mPer Phrasing:\033[0m")
        for phrasing, acc in results_dict["phrasing_accuracy"].items():
            print(f"  {phrasing}: {acc:.2%}")
        if results_dict["check_failure_counts"]:
            print("\n\033[93mFailed checks:\033[0m")
            for name, count in results_dict["check_failure_counts"].items():
                print(f"  {name}: {count}")

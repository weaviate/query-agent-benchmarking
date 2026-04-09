"""Backward-compatible re-exports from domain.models.

All models are now defined in query_agent_benchmarking.internal.core.models.
This module re-exports them for backward compatibility.
"""

from query_agent_benchmarking.internal.core.models import (
    ObjectID,
    NuggetInfo,
    InMemoryQuery,
    InMemorySearchQuery,
    InMemoryAskQuery,
    QueryResult,
    SearchResult,
    AskResult,
    DocsCollection,
    QueriesCollection,
    AskQueriesCollection,
    HardNegativesCollection,
)

__all__ = [
    "ObjectID",
    "NuggetInfo",
    "InMemoryQuery",
    "InMemorySearchQuery",
    "InMemoryAskQuery",
    "QueryResult",
    "SearchResult",
    "AskResult",
    "DocsCollection",
    "QueriesCollection",
    "AskQueriesCollection",
    "HardNegativesCollection",
]

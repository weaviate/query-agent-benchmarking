"""
Domain models for the query agent benchmarking system.

These Pydantic models define the core data contracts used throughout the system.
This module has no external infrastructure dependencies.
"""

from typing import Any, Optional
from pydantic import BaseModel


class ObjectID(BaseModel):
    object_id: str


# ============================================================================
# Query Models
# ============================================================================

class NuggetInfo(BaseModel):
    """Information about a single nugget (used by FreshStack)."""
    nugget_id: str
    text: str
    relevant_corpus_ids: list[str]


class InMemoryQuery(BaseModel):
    """Base query model - kept for backwards compatibility."""
    question: str
    dataset_ids: list[str]
    query_id: Optional[str] = None
    tenant_id: Optional[str] = None
    # FreshStack nugget fields (optional)
    nugget_data: Optional[list[NuggetInfo]] = None
    ids_per_nugget: Optional[dict[str, list[str]]] = None
    num_nuggets: Optional[int] = None


InMemorySearchQuery = InMemoryQuery


class InMemoryAskQuery(BaseModel):
    """Query model for ask benchmarks (question answering evaluation)."""
    question: str
    ground_truth_answer: str  # Ground truth answer for LLM judge comparison
    oracle_context_id: Optional[str] = None  # Optional: for oracle/hard-negative experiments
    tenant_id: Optional[str] = None  # Optional: for multi-tenant datasets (e.g., LongMemEval)
    question_type: Optional[str] = None  # Optional: for type-specific evaluation (e.g., LongMemEval)
    question_date: Optional[str] = None  # Optional: date the question was asked (e.g., LongMemEval)


# ============================================================================
# Result Models
# ============================================================================

class AgentSearch(BaseModel):
    """A single sub-query a search agent issued under the hood.

    Infrastructure-agnostic mirror of the Weaviate SDK's
    ``QueryResultWithCollectionNormalized``. It records *how* an agent
    decomposed one natural-language query into a structured search against a
    collection: an optional semantic query string, an optional (recursive)
    filter tree, an optional sort, and an optional direct UUID lookup.

    ``filters`` and ``sort_property`` are stored as JSON-serializable values
    (the SDK models dumped with ``mode="json"``) so the domain layer stays free
    of SDK types and the structures persist/serialize cleanly. ``filters`` is
    either a leaf filter (``{"filter_type", "property_name", "operator",
    "value", ...}``) or a boolean group (``{"combine": "AND"|"OR", "filters":
    [...]}``).
    """
    collection: str
    query: Optional[str] = None
    filters: Optional[Any] = None
    sort_property: Optional[Any] = None
    uuid_value: Optional[str] = None


class QueryResult(BaseModel):
    """Result from a search query - kept for backwards compatibility.

    ``searches`` is optional: ``None`` means the agent does not report a search
    plan (e.g. direct hybrid/vector/BM25 or an external service that hasn't
    implemented it), which is distinct from an empty list (a plan with zero
    sub-searches).
    """
    query: InMemoryQuery
    query_ground_truth_id: list[str]
    retrieved_ids: list[ObjectID]
    time_taken: float
    searches: Optional[list[AgentSearch]] = None


SearchResult = QueryResult


class SearchAgentResponse(BaseModel):
    """Optional richer return type for ``SearchAgent.run`` / ``run_async``.

    Agents may return a plain ``list[ObjectID]`` (the legacy contract, still
    fully supported for BYOS and direct-search adapters) or this object to
    additionally expose the structured ``searches`` the agent performed. The
    query-execution layer normalizes both shapes, so returning this is purely
    additive.

    ``searches`` is optional: leave it ``None`` to signal that this agent does
    not (yet) report a search plan, while still using the richer response type.
    """
    retrieved_ids: list[ObjectID]
    searches: Optional[list[AgentSearch]] = None


class AskResult(BaseModel):
    """Result from an ask benchmark query."""
    query: InMemoryAskQuery
    system_answer: str
    alignment_score: Optional[bool] = None  # Set after LLM judge evaluation
    time_taken: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    retrieved_context: Optional[Any] = None  # Raw context from the agent (e.g. memories)


# ============================================================================
# Collection Configuration Models
# ============================================================================

class DocsCollection(BaseModel):
    collection_name: str
    content_key: str
    id_key: str


class QueriesCollection(BaseModel):
    collection_name: str
    query_content_key: str
    gold_ids_key: str


class AskQueriesCollection(BaseModel):
    """Collection configuration for ask mode queries."""
    collection_name: str
    query_content_key: str  # e.g., "question"
    answer_key: str  # e.g., "answer" - the expected answer
    oracle_context_id_key: Optional[str] = None  # e.g., "dataset_id" - optional


# ============================================================================
# Hard Negatives Models
# ============================================================================

# WIP: Not sure this is the best way to store the hard negatives
# This tells the query-agent-benchmarking package how to store the hard negatives for each query
class HardNegativesCollection(BaseModel):
    collection_name: str
    query_content_key: str  # e.g., "query"
    gold_ids_key: str  # e.g., "gold_doc_ids" - stores list of gold IDs
    gold_documents_key: str  # e.g., "gold_documents" - stores list/text of gold doc contents
    hard_negative_document_key: str  # e.g., "hard_negative_doc" - stores the hard negative doc content
    hard_negative_id_key: str  # e.g., "hard_negative_id" - stores the hard negative doc ID

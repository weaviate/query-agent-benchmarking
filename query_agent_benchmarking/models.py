from typing import Optional
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


# ============================================================================
# Result Models
# ============================================================================

class QueryResult(BaseModel):
    """Result from a search query - kept for backwards compatibility."""
    query: InMemoryQuery
    query_ground_truth_id: list[str]
    retrieved_ids: list[ObjectID]
    time_taken: float


SearchResult = QueryResult


class AskResult(BaseModel):
    """Result from an ask benchmark query."""
    query: InMemoryAskQuery
    system_answer: str
    alignment_score: Optional[bool] = None  # Set after LLM judge evaluation
    time_taken: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


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
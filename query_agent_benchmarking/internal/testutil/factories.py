"""Factory functions for creating test data."""

from query_agent_benchmarking.internal.core.domain.models import (
    ObjectID,
    InMemoryQuery,
    InMemoryAskQuery,
    QueryResult,
    AskResult,
    NuggetInfo,
)


def make_search_queries() -> list[InMemoryQuery]:
    """Create sample search queries with ground truth document IDs."""
    return [
        InMemoryQuery(
            question="What is information retrieval?",
            dataset_ids=["doc1", "doc2", "doc3"],
        ),
        InMemoryQuery(
            question="Explain vector search",
            dataset_ids=["doc4", "doc5"],
        ),
        InMemoryQuery(
            question="What is BM25?",
            dataset_ids=["doc6"],
        ),
    ]


def make_search_results(queries: list[InMemoryQuery] | None = None) -> list[QueryResult]:
    """Create sample search results matching sample queries."""
    if queries is None:
        queries = make_search_queries()
    return [
        QueryResult(
            query=queries[0],
            query_ground_truth_id=["doc1", "doc2", "doc3"],
            retrieved_ids=[
                ObjectID(object_id="doc1"),
                ObjectID(object_id="doc7"),
                ObjectID(object_id="doc2"),
                ObjectID(object_id="doc8"),
                ObjectID(object_id="doc3"),
            ],
            time_taken=0.5,
        ),
        QueryResult(
            query=queries[1],
            query_ground_truth_id=["doc4", "doc5"],
            retrieved_ids=[
                ObjectID(object_id="doc4"),
                ObjectID(object_id="doc9"),
                ObjectID(object_id="doc5"),
            ],
            time_taken=0.3,
        ),
        QueryResult(
            query=queries[2],
            query_ground_truth_id=["doc6"],
            retrieved_ids=[
                ObjectID(object_id="doc10"),
                ObjectID(object_id="doc6"),
            ],
            time_taken=0.2,
        ),
    ]


def make_freshstack_query() -> InMemoryQuery:
    """Create a FreshStack query with nugget data."""
    return InMemoryQuery(
        question="How to use LangChain?",
        dataset_ids=["doc1", "doc2"],
        nugget_data=[
            NuggetInfo(
                nugget_id="n1",
                text="LangChain basics",
                relevant_corpus_ids=["doc1"],
            ),
            NuggetInfo(
                nugget_id="n2",
                text="LangChain chains",
                relevant_corpus_ids=["doc2", "doc3"],
            ),
            NuggetInfo(
                nugget_id="n3",
                text="LangChain agents",
                relevant_corpus_ids=["doc4"],
            ),
        ],
    )


def make_ask_queries() -> list[InMemoryAskQuery]:
    """Create sample ask queries with ground truth answers."""
    return [
        InMemoryAskQuery(
            question="What is HyDE?",
            ground_truth_answer="HyDE stands for Hypothetical Document Embeddings, a retrieval technique.",
        ),
        InMemoryAskQuery(
            question="What year was BERT released?",
            ground_truth_answer="2018",
        ),
    ]


def make_ask_results(queries: list[InMemoryAskQuery] | None = None) -> list[AskResult]:
    """Create sample ask results matching sample queries."""
    if queries is None:
        queries = make_ask_queries()
    return [
        AskResult(
            query=queries[0],
            system_answer="HyDE is Hypothetical Document Embeddings, used for retrieval.",
            time_taken=1.5,
        ),
        AskResult(
            query=queries[1],
            system_answer="BERT was released in 2018.",
            time_taken=0.8,
        ),
    ]

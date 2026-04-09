"""Shared test fixtures for query agent benchmarking tests."""

import pytest

from query_agent_benchmarking.domain.models import (
    ObjectID,
    InMemoryQuery,
    InMemoryAskQuery,
    QueryResult,
    AskResult,
    NuggetInfo,
)
from query_agent_benchmarking.ports.ask_agent import AskResponse


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_search_queries():
    """Basic search queries with ground truth document IDs."""
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


@pytest.fixture
def sample_search_results(sample_search_queries):
    """Search results matching sample queries."""
    return [
        QueryResult(
            query=sample_search_queries[0],
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
            query=sample_search_queries[1],
            query_ground_truth_id=["doc4", "doc5"],
            retrieved_ids=[
                ObjectID(object_id="doc4"),
                ObjectID(object_id="doc9"),
                ObjectID(object_id="doc5"),
            ],
            time_taken=0.3,
        ),
        QueryResult(
            query=sample_search_queries[2],
            query_ground_truth_id=["doc6"],
            retrieved_ids=[
                ObjectID(object_id="doc10"),
                ObjectID(object_id="doc6"),
            ],
            time_taken=0.2,
        ),
    ]


@pytest.fixture
def sample_freshstack_query():
    """A FreshStack query with nugget data."""
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


@pytest.fixture
def sample_ask_queries():
    """Basic ask queries with ground truth answers."""
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


@pytest.fixture
def sample_ask_results(sample_ask_queries):
    """Ask results matching sample queries."""
    return [
        AskResult(
            query=sample_ask_queries[0],
            system_answer="HyDE is Hypothetical Document Embeddings, used for retrieval.",
            time_taken=1.5,
        ),
        AskResult(
            query=sample_ask_queries[1],
            system_answer="BERT was released in 2018.",
            time_taken=0.8,
        ),
    ]


# ============================================================================
# Mock Agent Fixtures
# ============================================================================

class MockSearchAgent:
    """Mock search agent for testing."""

    def __init__(self, responses: dict[str, list[str]] | None = None):
        self.responses = responses or {}
        self.call_count = 0

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        self.call_count += 1
        ids = self.responses.get(query, [])
        return [ObjectID(object_id=id_) for id_ in ids]

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        return self.run(query, tenant)

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass


class MockAskAgent:
    """Mock ask agent for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0

    def run(self, query: str, oracle_context_id=None, tenant_id=None) -> AskResponse:
        self.call_count += 1
        answer = self.responses.get(query, "I don't know.")
        return AskResponse(final_answer=answer)

    async def run_async(self, query: str, oracle_context_id=None, tenant_id=None) -> AskResponse:
        return self.run(query, oracle_context_id, tenant_id)

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass


class MockResultRepository:
    """Mock result repository for testing."""

    def __init__(self):
        self.saved_trials = []
        self.saved_ask_trials = []
        self.saved_metrics = []
        self.saved_aggregated = []

    def save_trial_results(self, results, config, trial_number):
        self.saved_trials.append((results, config, trial_number))

    def save_ask_trial_results(self, results, config, trial_number, alignment_scores=None):
        self.saved_ask_trials.append((results, config, trial_number, alignment_scores))

    def save_trial_metrics(self, metrics, config, trial_number):
        self.saved_metrics.append((metrics, config, trial_number))

    def save_aggregated_results(self, aggregated_metrics, config):
        self.saved_aggregated.append((aggregated_metrics, config))


@pytest.fixture
def mock_search_agent():
    return MockSearchAgent(responses={
        "What is information retrieval?": ["doc1", "doc2", "doc7"],
        "Explain vector search": ["doc4", "doc9"],
        "What is BM25?": ["doc10", "doc6"],
    })


@pytest.fixture
def mock_ask_agent():
    return MockAskAgent(responses={
        "What is HyDE?": "HyDE is Hypothetical Document Embeddings for retrieval.",
        "What year was BERT released?": "BERT was released in 2018.",
    })


@pytest.fixture
def mock_result_repository():
    return MockResultRepository()

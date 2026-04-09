"""Shared test fixtures for query agent benchmarking tests."""

import pytest

from query_agent_benchmarking.internal.mocks import (
    MockSearchAgent,
    MockAskAgent,
    MockResultRepository,
)
from query_agent_benchmarking.internal.testutil import (
    make_search_queries,
    make_search_results,
    make_freshstack_query,
    make_ask_queries,
    make_ask_results,
)


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_search_queries():
    """Basic search queries with ground truth document IDs."""
    return make_search_queries()


@pytest.fixture
def sample_search_results(sample_search_queries):
    """Search results matching sample queries."""
    return make_search_results(sample_search_queries)


@pytest.fixture
def sample_freshstack_query():
    """A FreshStack query with nugget data."""
    return make_freshstack_query()


@pytest.fixture
def sample_ask_queries():
    """Basic ask queries with ground truth answers."""
    return make_ask_queries()


@pytest.fixture
def sample_ask_results(sample_ask_queries):
    """Ask results matching sample queries."""
    return make_ask_results(sample_ask_queries)


# ============================================================================
# Mock Agent Fixtures
# ============================================================================

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

"""Functional tests for dataset loading.

These tests download real data from HuggingFace Hub and ir_datasets to verify
that every supported dataset can be loaded and produces well-formed output.

Run with:
    uv run pytest tests/functional/ -v
    uv run pytest tests/functional/ -v -k "beir"       # just BEIR datasets
    uv run pytest tests/functional/ -v -k "freshstack"  # just FreshStack
"""

import pytest

from query_agent_benchmarking.internal.adapters.dataset import (
    in_memory_dataset_loader,
    in_memory_ask_dataset_loader,
)
from query_agent_benchmarking.internal.core.domain.models import InMemoryQuery, InMemoryAskQuery


# ---------------------------------------------------------------------------
# Search datasets — one representative per loader family
# ---------------------------------------------------------------------------

SEARCH_DATASETS = [
    # ir_datasets loader (BEIR)
    "beir/scifact/test",
    "beir/fiqa/test",
    "beir/nq",
    # ir_datasets loader (LoTTe)
    "lotte/lifestyle/test/search",
    # HuggingFace loader (BRIGHT)
    "bright/biology",
    # HuggingFace loader (FreshStack)
    "freshstack-langchain",
    # HuggingFace loader (standalone)
    "enron",
    "wixqa",
    "irpapers",
    "multihoprag",
    "longmemeval-s",
    # officeqa and vidore_v3_hr are excluded:
    #   - officeqa requires local PDF files
    #   - vidore_v3_hr downloads very large image data
]


ASK_DATASETS = [
    "irpapers",
    "multihoprag",
    "longmemeval-s",
    # officeqa excluded — requires local CSV/PDF files
]


# ---------------------------------------------------------------------------
# Search dataset tests
# ---------------------------------------------------------------------------

class TestSearchDatasetLoading:
    """Verify each search dataset loads at least one doc and one query."""

    @pytest.mark.parametrize("dataset_name", SEARCH_DATASETS)
    def test_load_corpus(self, dataset_name):
        """Loading corpus_only returns at least one document with a dataset_id."""
        docs, queries = in_memory_dataset_loader(dataset_name, corpus_only=True)

        assert len(docs) > 0, f"{dataset_name}: corpus is empty"
        assert queries == [], f"{dataset_name}: queries should be empty with corpus_only=True"

        first_doc = docs[0]
        assert isinstance(first_doc, dict), f"{dataset_name}: doc is not a dict"
        id_fields = {"dataset_id", "doc_id", "id", "corpus_id"}
        assert id_fields & set(first_doc.keys()), (
            f"{dataset_name}: doc missing identifier field. Keys: {list(first_doc.keys())}"
        )

    @pytest.mark.parametrize("dataset_name", SEARCH_DATASETS)
    def test_load_queries(self, dataset_name):
        """Loading queries_only returns at least one well-formed InMemoryQuery."""
        docs, queries = in_memory_dataset_loader(dataset_name, queries_only=True)

        assert docs == [], f"{dataset_name}: docs should be empty with queries_only=True"
        assert len(queries) > 0, f"{dataset_name}: queries list is empty"

        first_query = queries[0]
        assert isinstance(first_query, InMemoryQuery), (
            f"{dataset_name}: expected InMemoryQuery, got {type(first_query).__name__}"
        )
        assert first_query.question, f"{dataset_name}: query has empty question"
        assert isinstance(first_query.dataset_ids, list), (
            f"{dataset_name}: dataset_ids is not a list"
        )

    @pytest.mark.parametrize("dataset_name", SEARCH_DATASETS)
    def test_load_full(self, dataset_name):
        """Loading without flags returns both docs and queries."""
        docs, queries = in_memory_dataset_loader(dataset_name)

        assert len(docs) > 0, f"{dataset_name}: no docs"
        assert len(queries) > 0, f"{dataset_name}: no queries"


# ---------------------------------------------------------------------------
# Ask dataset tests
# ---------------------------------------------------------------------------

class TestAskDatasetLoading:
    """Verify each ask dataset loads at least one well-formed query."""

    @pytest.mark.parametrize("dataset_name", ASK_DATASETS)
    def test_load_ask_queries(self, dataset_name):
        """Each ask dataset returns at least one InMemoryAskQuery."""
        queries = in_memory_ask_dataset_loader(dataset_name)

        assert len(queries) > 0, f"{dataset_name}: ask queries list is empty"

        first = queries[0]
        assert isinstance(first, InMemoryAskQuery), (
            f"{dataset_name}: expected InMemoryAskQuery, got {type(first).__name__}"
        )
        assert first.question, f"{dataset_name}: ask query has empty question"
        assert first.ground_truth_answer, (
            f"{dataset_name}: ask query has empty ground_truth_answer"
        )

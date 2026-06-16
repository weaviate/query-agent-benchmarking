"""Functional tests that load data into a live Weaviate instance.

These tests require WEAVIATE_URL and WEAVIATE_API_KEY to be set.
They create temporary collections with a "_FuncTest" tag, insert a small
sample of documents, verify the data landed, then clean up.

Run with:
    uv run pytest tests/functional/test_database_weaviate.py -v
    uv run pytest tests/functional/test_database_weaviate.py -v -k "enron"
"""

import os

import pytest

from query_agent_benchmarking.internal.adapters.dataset import in_memory_dataset_loader
from query_agent_benchmarking.internal.adapters.database.database_registry import resolve_spec
from query_agent_benchmarking.internal.adapters.database.database_loader import (
    _drop_and_create_collection,
    _batch_insert,
    get_vector_config,
)
from query_agent_benchmarking.internal.adapters.database.naming import add_tag_to_name
from query_agent_benchmarking.internal.adapters.clients.weaviate_client import get_weaviate_client


# Skip entire module if Weaviate credentials are not set.
pytestmark = pytest.mark.skipif(
    not os.getenv("WEAVIATE_URL") or not os.getenv("WEAVIATE_API_KEY"),
    reason="WEAVIATE_URL and WEAVIATE_API_KEY required",
)

TAG = "FuncTest"
SAMPLE_SIZE = 5

# Datasets to test. We use a subset to keep it fast.
# Multi-tenant (longmemeval) is tested separately.
STANDARD_DATASETS = [
    "beir/scifact/test",
    "bright/biology",
    "enron",
    "freshstack-langchain",
    "wixqa",
    "irpapers-text-only",
    "filtered-cars",
    "multihoprag",
    "lotte/lifestyle/test/search",
]

MULTI_TENANT_DATASETS = [
    "longmemeval-s",
]


@pytest.fixture(scope="module")
def weaviate_client():
    """Shared Weaviate client for the entire test module."""
    client = get_weaviate_client()
    yield client
    client.close()


def _collection_name(dataset_name: str) -> str:
    spec = resolve_spec(dataset_name)
    return add_tag_to_name(spec.name_fn(dataset_name), TAG)


def _cleanup(client, name):
    """Delete collection if it exists."""
    if client.collections.exists(name):
        client.collections.delete(name)


class TestStandardDatabaseLoading:
    """Load a small sample into Weaviate and verify it."""

    @pytest.mark.parametrize("dataset_name", STANDARD_DATASETS)
    def test_create_and_insert(self, weaviate_client, dataset_name):
        """Create collection, insert sample docs, verify count, then clean up."""
        spec = resolve_spec(dataset_name)
        collection_name = _collection_name(dataset_name)

        # Clean up from any prior failed run
        _cleanup(weaviate_client, collection_name)

        try:
            # Load a small sample of documents
            docs, _ = in_memory_dataset_loader(dataset_name, corpus_only=True)
            sample = docs[:SAMPLE_SIZE]

            # Create collection with default vectorizer
            _drop_and_create_collection(
                weaviate_client,
                collection_name,
                properties=spec.properties,
                vector_config=get_vector_config(),
                recreate=True,
            )

            assert weaviate_client.collections.exists(collection_name), (
                f"{dataset_name}: collection was not created"
            )

            # Insert sample
            inserted = _batch_insert(
                weaviate_client,
                collection=collection_name,
                items=sample,
                item_to_props=spec.item_to_props,
                batch_size=SAMPLE_SIZE,
                verbose=False,
            )

            assert inserted == SAMPLE_SIZE, (
                f"{dataset_name}: expected {SAMPLE_SIZE} inserts, got {inserted}"
            )

            # Verify objects are queryable
            collection = weaviate_client.collections.get(collection_name)
            response = collection.aggregate.over_all(total_count=True)
            assert response.total_count == SAMPLE_SIZE, (
                f"{dataset_name}: expected {SAMPLE_SIZE} objects in Weaviate, "
                f"got {response.total_count}"
            )

        finally:
            _cleanup(weaviate_client, collection_name)


class TestMultiTenantDatabaseLoading:
    """Load multi-tenant data into Weaviate and verify it."""

    @pytest.mark.parametrize("dataset_name", MULTI_TENANT_DATASETS)
    def test_create_and_insert_multi_tenant(self, weaviate_client, dataset_name):
        """Create multi-tenant collection, insert sample, verify, then clean up."""
        spec = resolve_spec(dataset_name)
        collection_name = _collection_name(dataset_name)

        _cleanup(weaviate_client, collection_name)

        try:
            docs, _ = in_memory_dataset_loader(dataset_name, corpus_only=True)
            sample = docs[:SAMPLE_SIZE]

            _drop_and_create_collection(
                weaviate_client,
                collection_name,
                properties=spec.properties,
                vector_config=get_vector_config(),
                recreate=True,
                multi_tenancy_config=spec.multi_tenancy_config,
            )

            assert weaviate_client.collections.exists(collection_name)

            inserted = _batch_insert(
                weaviate_client,
                collection=collection_name,
                items=sample,
                item_to_props=spec.item_to_props,
                batch_size=SAMPLE_SIZE,
                verbose=False,
                tenant_id_field=spec.tenant_id_field,
            )

            assert inserted == SAMPLE_SIZE, (
                f"{dataset_name}: expected {SAMPLE_SIZE} inserts, got {inserted}"
            )

        finally:
            _cleanup(weaviate_client, collection_name)

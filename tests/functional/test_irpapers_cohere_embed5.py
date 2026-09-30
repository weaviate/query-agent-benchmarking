"""Functional test: populate Weaviate with IRPAPERS page images vectorized by Cohere Embed 5.

Builds an IRPAPERS collection whose only vector is `image_content_cohere`, a
multi2vec-cohere vector over the page image blob, inserts a sample of pages,
and verifies every object was vectorized and is searchable by text.

Requires WEAVIATE_URL, WEAVIATE_API_KEY and COHERE_API_KEY.

Run with:
    uv run pytest tests/functional/test_irpapers_cohere_embed5.py -v
    uv run pytest tests/functional/test_irpapers_cohere_embed5.py -v -k "pro"

Environment knobs:
    IRPAPERS_EMBED5_SAMPLE_SIZE  Number of pages to insert (default 5, "all" for the full corpus)
    IRPAPERS_EMBED5_KEEP=1       Keep the collection after the test instead of deleting it
"""

import os

import pytest

from query_agent_benchmarking.internal.adapters.dataset import in_memory_dataset_loader
from query_agent_benchmarking.internal.adapters.database.property_builder import DatasetSpecBuilder
from query_agent_benchmarking.internal.adapters.database.database_loader import (
    _drop_and_create_collection,
    _batch_insert,
    _resolve_provider_headers,
)
from query_agent_benchmarking.internal.adapters.database.naming import add_tag_to_name
from query_agent_benchmarking.internal.adapters.clients.weaviate_client import get_weaviate_client


pytestmark = pytest.mark.skipif(
    not os.getenv("WEAVIATE_URL")
    or not os.getenv("WEAVIATE_API_KEY")
    or not os.getenv("COHERE_API_KEY"),
    reason="WEAVIATE_URL, WEAVIATE_API_KEY and COHERE_API_KEY required",
)

EMBED5_MODELS = [
    "cohere/embed-v5.0-pro",
    "cohere/embed-v5.0-fast",
]
IMAGE_VECTOR_NAME = "image_content_cohere"

_raw_sample_size = os.getenv("IRPAPERS_EMBED5_SAMPLE_SIZE", "5")
SAMPLE_SIZE = None if _raw_sample_size.lower() == "all" else int(_raw_sample_size)
KEEP_COLLECTION = os.getenv("IRPAPERS_EMBED5_KEEP") == "1"


def _build_irpapers_embed5_spec(embedding_model: str):
    """IRPAPERS spec with a single Cohere Embed 5 image vector (no text vector)."""
    return (
        DatasetSpecBuilder("irpapers")
        .with_static_name("IRPapers")
        .with_text_property("content", source_field="transcription")
        .with_blob_property("image", source_field="base64_str")
        .with_dataset_id()
        .with_multi2vec(
            image_field="image",
            name="image_content",
            name_by_provider=True,
            embedding_model=embedding_model,
        )
        .build()
    )


def _tag_for(embedding_model: str) -> str:
    # "cohere/embed-v5.0-pro" -> "FuncTestEmbed5Pro"
    variant = embedding_model.rsplit("-", 1)[-1].capitalize()
    return f"FuncTestEmbed5{variant}"


@pytest.fixture(scope="module")
def irpapers_docs():
    docs, _ = in_memory_dataset_loader("irpapers", corpus_only=True)
    return docs if SAMPLE_SIZE is None else docs[:SAMPLE_SIZE]


@pytest.fixture(scope="module")
def weaviate_client():
    headers = _resolve_provider_headers(embedding_models=EMBED5_MODELS)
    client = get_weaviate_client(headers=headers)
    yield client
    client.close()


@pytest.mark.parametrize("embedding_model", EMBED5_MODELS, ids=lambda m: m.rsplit("-", 1)[-1])
def test_populate_irpapers_with_embed5_image_vectors(weaviate_client, irpapers_docs, embedding_model):
    """Create collection, insert page images, verify each has an Embed 5 vector, run a text query."""
    spec = _build_irpapers_embed5_spec(embedding_model)
    collection_name = add_tag_to_name(spec.name_fn("irpapers"), _tag_for(embedding_model))

    try:
        _drop_and_create_collection(
            weaviate_client,
            collection_name,
            properties=spec.properties,
            vector_config=spec.vector_config,
            recreate=True,
        )

        _batch_insert(
            weaviate_client,
            collection=collection_name,
            items=irpapers_docs,
            item_to_props=spec.item_to_props,
            batch_size=min(20, len(irpapers_docs)),
            verbose=SAMPLE_SIZE is None,
        )

        # _batch_insert counts queued objects; vectorizer errors only surface here.
        failed = weaviate_client.batch.failed_objects
        assert not failed, (
            f"{embedding_model}: {len(failed)} objects failed, first error: {failed[0].message}"
        )

        collection = weaviate_client.collections.get(collection_name)
        total = collection.aggregate.over_all(total_count=True).total_count
        assert total == len(irpapers_docs), (
            f"{embedding_model}: expected {len(irpapers_docs)} objects, got {total}"
        )

        # Every object should carry a non-empty Embed 5 image vector.
        dims = set()
        for obj in collection.iterator(include_vector=[IMAGE_VECTOR_NAME]):
            vector = obj.vector.get(IMAGE_VECTOR_NAME)
            assert vector, f"{embedding_model}: object {obj.uuid} has no {IMAGE_VECTOR_NAME} vector"
            dims.add(len(vector))
        assert len(dims) == 1, f"{embedding_model}: inconsistent vector dimensions {dims}"

        # Cross-modal sanity check: a text query against the image vector returns pages.
        response = collection.query.near_text(
            query="retrieval evaluation results table",
            target_vector=IMAGE_VECTOR_NAME,
            limit=3,
            return_properties=["dataset_id"],
        )
        assert response.objects, f"{embedding_model}: near_text returned no results"
        print(
            f"\n{embedding_model}: {total} pages, dim={dims.pop()}, "
            f"top hits={[o.properties['dataset_id'] for o in response.objects]}"
        )

    finally:
        if not KEEP_COLLECTION and weaviate_client.collections.exists(collection_name):
            weaviate_client.collections.delete(collection_name)

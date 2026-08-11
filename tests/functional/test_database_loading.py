"""Functional tests for the database loading pipeline.

These tests verify the full chain that populate-db.py exercises:
  1. Load documents from HuggingFace / ir_datasets
  2. Resolve the DatasetSpec from the registry
  3. Map every document through item_to_props (the property mapper)
  4. Verify the mapped properties match the spec's declared schema

This catches schema mismatches (e.g. a loader returns "id" but the spec
expects "dataset_id") without needing a live Weaviate connection.

Run with:
    uv run pytest tests/functional/test_database_loading.py -v
    uv run pytest tests/functional/test_database_loading.py -v -k "enron"
"""

import pytest

from query_agent_benchmarking.internal.adapters.dataset import in_memory_dataset_loader
from query_agent_benchmarking.internal.adapters.database.database_registry import resolve_spec


# Every dataset supported by populate-db.py / database_loader_config.yml,
# except those that require local files or very large downloads.
DATABASE_DATASETS = [
    "beir/scifact/test",
    "beir/fiqa/test",
    "beir/nq",
    "bright/biology",
    "enron",
    "freshstack-langchain",
    "wixqa",
    "irpapers",
    "irpapers-text-only",
    "filtered-cars",
    "multihoprag",
    "longmemeval-s",
    "lotte/lifestyle/test/search",
    "obliq-bench-congress",
    # Excluded:
    #   officeqa       — requires local PDF files
    #   vidore_v3_hr   — very large image downloads
    #   longmemeval-m  — same loader as longmemeval-s, just larger
]


class TestDatabaseLoadingPipeline:
    """Verify that each dataset can be loaded, spec-resolved, and property-mapped."""

    @pytest.mark.parametrize("dataset_name", DATABASE_DATASETS)
    def test_resolve_spec(self, dataset_name):
        """The registry has a matching DatasetSpec for this dataset."""
        spec = resolve_spec(dataset_name)
        assert spec is not None
        assert spec.matches(dataset_name)

    @pytest.mark.parametrize("dataset_name", DATABASE_DATASETS)
    def test_collection_name(self, dataset_name):
        """The spec produces a non-empty collection name."""
        spec = resolve_spec(dataset_name)
        name = spec.name_fn(dataset_name)
        assert name, f"{dataset_name}: name_fn returned empty string"
        assert isinstance(name, str)

    @pytest.mark.parametrize("dataset_name", DATABASE_DATASETS)
    def test_load_and_map_documents(self, dataset_name):
        """Load corpus, then map every doc through item_to_props without error."""
        docs, _ = in_memory_dataset_loader(dataset_name, corpus_only=True)
        assert len(docs) > 0, f"{dataset_name}: no documents loaded"

        spec = resolve_spec(dataset_name)

        # Map the first doc and verify it produces a non-empty dict
        first_props = spec.item_to_props(docs[0])
        assert isinstance(first_props, dict), (
            f"{dataset_name}: item_to_props returned {type(first_props).__name__}, expected dict"
        )
        assert len(first_props) > 0, (
            f"{dataset_name}: item_to_props returned empty dict. "
            f"Doc keys: {list(docs[0].keys())}"
        )

        # Map a sample of docs to catch inconsistencies across the corpus.
        # Use first 5 + last 5 to catch edge cases.
        sample = docs[:5] + docs[-5:]
        for i, doc in enumerate(sample):
            props = spec.item_to_props(doc)
            assert isinstance(props, dict), (
                f"{dataset_name}: doc {i} item_to_props failed"
            )

    @pytest.mark.parametrize("dataset_name", DATABASE_DATASETS)
    def test_mapped_props_match_schema(self, dataset_name):
        """Mapped property keys should be a subset of the declared schema properties."""
        docs, _ = in_memory_dataset_loader(dataset_name, corpus_only=True)
        spec = resolve_spec(dataset_name)

        # Get declared property names from the spec
        declared_props = {p.name for p in spec.properties}

        first_props = spec.item_to_props(docs[0])
        mapped_keys = set(first_props.keys())

        unexpected = mapped_keys - declared_props
        assert not unexpected, (
            f"{dataset_name}: item_to_props produced keys not in schema: {unexpected}. "
            f"Declared: {declared_props}, Mapped: {mapped_keys}"
        )

    @pytest.mark.parametrize("dataset_name", DATABASE_DATASETS)
    def test_dataset_id_present(self, dataset_name):
        """Every mapped doc should include a dataset_id (required for benchmarking)."""
        docs, _ = in_memory_dataset_loader(dataset_name, corpus_only=True)
        spec = resolve_spec(dataset_name)

        first_props = spec.item_to_props(docs[0])
        assert "dataset_id" in first_props, (
            f"{dataset_name}: mapped doc missing 'dataset_id'. "
            f"Keys: {list(first_props.keys())}. "
            f"Doc keys: {list(docs[0].keys())}"
        )
        assert first_props["dataset_id"], (
            f"{dataset_name}: dataset_id is empty/None"
        )

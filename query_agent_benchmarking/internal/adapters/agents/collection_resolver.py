"""Dataset-to-collection name mapping.

Extracted from agent/base.py to decouple collection naming from
agent connection management.
"""

from typing import Optional

from query_agent_benchmarking.internal.core.domain.models import DocsCollection
from query_agent_benchmarking.internal.adapters.database.database_registry import resolve_spec


def resolve_collection_info(
    dataset_name: Optional[str] = None,
    docs_collection: Optional[DocsCollection] = None,
) -> dict:
    """Resolve collection name and metadata from dataset or custom collection config.

    Args:
        dataset_name: Built-in dataset name (e.g., "beir/scifact/test").
        docs_collection: Custom collection configuration.

    Returns:
        Dictionary with 'collection', 'id_property', 'uses_named_vectors',
        and 'available_named_vectors' keys.

    Raises:
        ValueError: If both or neither of dataset_name and docs_collection are provided.
    """
    if dataset_name and docs_collection:
        raise ValueError("Cannot specify both dataset_name and docs_collection")
    if not dataset_name and not docs_collection:
        raise ValueError("Must specify either dataset_name or docs_collection")

    if docs_collection:
        return {
            "collection": docs_collection.collection_name,
            "id_property": docs_collection.id_key,
            "uses_named_vectors": False,
            "available_named_vectors": [],
        }

    spec = resolve_spec(dataset_name)
    uses_named_vectors = isinstance(spec.vector_config, list)
    available_named_vectors = []
    if uses_named_vectors:
        available_named_vectors = [
            cfg.name
            for cfg in spec.vector_config
            if getattr(cfg, "name", None)
        ]

    return {
        "collection": f"{spec.name_fn(dataset_name)}_Default",
        "id_property": "dataset_id",
        "uses_named_vectors": uses_named_vectors,
        "available_named_vectors": available_named_vectors,
    }

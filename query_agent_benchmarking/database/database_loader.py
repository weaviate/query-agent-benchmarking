"""Database loading and collection management utilities."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import weaviate
import weaviate.collections.classes.config as wvcc

from .database_config import validate_database_dataset
from .database_registry import resolve_spec
from .engram_loader import engram_ingest_all_tenants
from ..dataset import in_memory_dataset_loader, load_longmemeval_docs_by_tenant
from ..utils import (
    get_weaviate_client,
    get_provider_headers,
    load_config,
    parse_embedding_model,
    pretty_print_in_memory_document,
    add_tag_to_name,
)


# Public API: used by external callers to build temporary collections with an explicit text embedding model.
def get_vector_config(embedding_model: Optional[str] = None) -> Any:
    """
    Factory function to create vectorizer config based on provider.
    
    Args:
        embedding_model: Model string in format "provider/model" (e.g., "cohere/embed-4")
                        or just "model" for weaviate. If None, uses default weaviate.
    
    Returns:
        Vectorizer configuration object
    """
    if not embedding_model:
        return wvcc.Configure.Vectors.text2vec_weaviate(
            vectorize_collection_name=False,
        )
    
    provider, model_name = parse_embedding_model(embedding_model)
    
    if provider == "weaviate":
        return wvcc.Configure.Vectors.text2vec_weaviate(
            model=model_name,
            vectorize_collection_name=False,
        )
    elif provider == "cohere":
        return wvcc.Configure.Vectors.text2vec_cohere(model=model_name)
    elif provider == "voyageai":
        return wvcc.Configure.Vectors.text2vec_voyageai(model=model_name)
    elif provider == "google":
        return wvcc.Configure.Vectors.multi2vec_google_gemini(model=model_name)
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported providers: ['weaviate', 'cohere', 'voyageai', 'google']"
        )


# Public API: used by embedding-comparison flows and scripts to create/populate tagged collections.
def create_collection_with_vector_config(
    client: Optional[weaviate.WeaviateClient] = None,
    dataset_name: Optional[str] = None,
    tag: str = "Default",
    embedding_model: Optional[str] = None,
    weaviate_client: Optional[weaviate.WeaviateClient] = None,
) -> None:
    """
    Create and populate a collection with a specified embedding model.
    
    Used for embedding model comparison where temporary collections
    are created with different models.
    
    Args:
        client: Connected Weaviate client
        dataset_name: Name of the dataset to load
        tag: Suffix to add to the collection name
        embedding_model: Embedding model to use. If None, uses default.
        weaviate_client: Backward-compatible alias for `client`.
    """
    if client is None:
        client = weaviate_client
    elif weaviate_client is not None and weaviate_client is not client:
        raise ValueError("Pass either `client` or `weaviate_client`, not both")

    if client is None:
        raise ValueError("A Weaviate client is required")
    if dataset_name is None:
        raise ValueError("dataset_name is required")

    print(f"Loading dataset '{dataset_name}'...")
    objects = _load_documents(dataset_name)

    spec = resolve_spec(dataset_name)
    alias_collection_name = spec.name_fn(dataset_name)
    collection_name = add_tag_to_name(alias_collection_name, tag)
    vector_config = get_vector_config(embedding_model)

    model_info = f" with model {embedding_model}" if embedding_model else " with default model"
    print(f"Creating collection '{collection_name}'{model_info}...")

    _drop_and_create_collection(
        client,
        collection_name,
        properties=spec.properties,
        vector_config=vector_config,
        recreate=True,
        multi_tenancy_config=spec.multi_tenancy_config,
    )

    print(f"Populating collection with {len(objects)} objects...")
    _batch_insert(
        client,
        collection=collection_name,
        items=objects,
        item_to_props=spec.item_to_props,
        tenant_id_field=spec.tenant_id_field,
    )
    print(f"Collection '{collection_name}' ready!\n")


# Public API: primary entry point used by `scripts/populate-db.py` and package users.
def database_loader(recreate: bool = True, tag: str = "Default") -> None:
    """
    Load dataset from config and populate the target database.

    Reads ``database_loader_config.yml`` to determine the dataset and target.
    When ``database_target`` is ``"engram"``, sessions are ingested into Engram
    (no Weaviate connection required). Otherwise defaults to Weaviate.

    Args:
        recreate: Whether to drop existing collection before creating
        tag: Suffix to add to collection name
    """
    config_path = Path(__file__).parent / "database_loader_config.yml"
    config = load_config(config_path)

    dataset_name: str = config["dataset_name"]
    validate_database_dataset(dataset_name)
    database_target = config.get("database_target", "weaviate")

    if database_target == "engram":
        _run_engram_loader(config, dataset_name)
        return

    _run_weaviate_loader(config, dataset_name, recreate=recreate, tag=tag)


def _run_engram_loader(config: dict, dataset_name: str) -> None:
    """Ingest LongMemEval sessions into Engram."""
    subset_cfg = config.get("longmemeval_subset")
    users_to_test = subset_cfg.get("users_to_test") if subset_cfg else None

    docs_by_tenant = load_longmemeval_docs_by_tenant(
        dataset_name,
        users_to_test=users_to_test,
    )

    engram_ingest_all_tenants(docs_by_tenant)


def _run_weaviate_loader(
    config: dict,
    dataset_name: str,
    recreate: bool = True,
    tag: str = "Default",
) -> None:
    """Load dataset into a Weaviate collection."""
    # Get provider headers for all configured embedding providers.
    headers = _resolve_provider_headers(
        embedding_providers=config.get("embedding_providers"),
        embedding_models=[
            config.get("text_embedding_model"),
            config.get("image_embedding_model"),
            *_coerce_model_list(config.get("text_embedding_models")),
            *_coerce_model_list(config.get("image_embedding_models")),
        ],
    )

    client = get_weaviate_client(headers=headers)

    try:
        objects = _load_documents(dataset_name)

        print("\033[92mFirst Document:\033[0m")
        pretty_print_in_memory_document(objects[0])

        spec = resolve_spec(dataset_name)
        alias_collection_name = spec.name_fn(dataset_name)
        collection_name = add_tag_to_name(alias_collection_name, tag)

        print(f"\n\033[96mCreating collection '{collection_name}'...\033[0m")
        _drop_and_create_collection(
            client,
            collection_name,
            properties=spec.properties,
            vector_config=spec.vector_config,
            recreate=recreate,
            multi_tenancy_config=spec.multi_tenancy_config,
        )

        # Manage alias
        alias_info = client.alias.get(alias_name=alias_collection_name)
        if alias_info is None:
            client.alias.create(
                alias_name=alias_collection_name,
                target_collection=collection_name,
            )
        else:
            client.alias.update(
                alias_name=alias_collection_name,
                new_target_collection=collection_name,
            )

        _batch_insert(
            client,
            collection=collection_name,
            items=objects,
            item_to_props=spec.item_to_props,
            tenant_id_field=spec.tenant_id_field,
        )
    finally:
        client.close()


# Private helper: low-level create/delete wrapper used internally to keep public loaders concise.
def _drop_and_create_collection(
    client: weaviate.WeaviateClient,
    name: str,
    properties: Sequence[wvcc.Property],
    vector_config: Any,
    recreate: bool = True,
    multi_tenancy_config: Any = None,
) -> None:
    """Drop (if exists) and create a Weaviate collection."""
    if recreate and client.collections.exists(name):
        client.collections.delete(name)
    if not client.collections.exists(name):
        create_kwargs: dict[str, Any] = {
            "name": name,
            "vector_config": vector_config,
            "properties": list(properties),
        }
        if multi_tenancy_config is not None:
            create_kwargs["multi_tenancy_config"] = multi_tenancy_config
        client.collections.create(**create_kwargs)


# Private helper: internal batching utility shared by the public loaders above.
def _batch_insert(
    client: weaviate.WeaviateClient,
    collection: str,
    items: Sequence[Mapping[str, Any]],
    item_to_props: Callable[[Mapping[str, Any]], dict[str, Any]],
    batch_size: int = 20,
    verbose: bool = True,
    tenant_id_field: Optional[str] = None,
) -> int:
    """
    Insert items into a Weaviate collection in batches.

    If tenant_id_field is set, each item's tenant is read from that field
    and passed to add_object so the object is inserted into the correct tenant.

    Returns the total number of items inserted.
    """
    start = time.perf_counter()
    total = 0

    if verbose:
        print(f"Inserting {len(items)} objects into collection '{collection}'...")

    with client.batch.fixed_size(batch_size=batch_size) as batch:
        for i, item in enumerate(items, start=1):
            props = item_to_props(item)
            add_kwargs: dict[str, Any] = {
                "collection": collection,
                "properties": props,
            }
            if tenant_id_field is not None:
                add_kwargs["tenant"] = str(item[tenant_id_field])
            batch.add_object(**add_kwargs)

            if verbose and i % batch_size == 0:
                elapsed = time.perf_counter() - start
                rate = i / max(elapsed, 1e-9)
                print(f"\033[92mInserted {i} objects ({elapsed:.1f}s, {rate:.1f} objs/s)\033[0m")

            total = i

    if verbose:
        elapsed = time.perf_counter() - start
        rate = total / max(elapsed, 1e-9)
        print(f"Inserted {total} objects in {elapsed:.2f}s ({rate:.1f} objs/s)")

    return total


# Private helper: internal header resolver so provider-key logic stays out of public APIs.
def _resolve_provider_headers(
    embedding_models: Sequence[Optional[str]] = (),
    embedding_providers: Optional[str | Sequence[str]] = None,
) -> dict[str, str]:
    """Resolve API headers for all configured embedding providers."""
    headers: dict[str, str] = {}

    for provider in _parse_embedding_providers(embedding_providers):
        headers.update(get_provider_headers(provider))

    for embedding_model in embedding_models:
        if not embedding_model:
            continue
        provider, _ = parse_embedding_model(embedding_model)
        headers.update(get_provider_headers(_normalize_provider_name(provider)))
    return headers


def _parse_embedding_providers(
    embedding_providers: Optional[str | Sequence[str]],
) -> list[str]:
    """Parse embedding providers from config values."""
    if embedding_providers is None:
        return []

    if isinstance(embedding_providers, str):
        raw = embedding_providers.strip()
        if not raw or raw.lower() == "auto":
            return []
        return [
            _normalize_provider_name(token)
            for token in raw.split(",")
            if token.strip()
        ]

    providers: list[str] = []
    for value in embedding_providers:
        if not isinstance(value, str):
            raise ValueError("embedding_providers entries must be strings.")
        normalized = _normalize_provider_name(value)
        if normalized == "auto":
            continue
        if normalized not in providers:
            providers.append(normalized)
    return providers


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider aliases to canonical provider names."""
    normalized = provider.strip().lower()
    if normalized in {"voyage", "voyage-ai"}:
        return "voyageai"
    if normalized in {"gemini", "google-gemini", "google_gemini"}:
        return "google"
    return normalized


def _coerce_model_list(raw_models: Any) -> list[str]:
    """Normalize optional model config values to a list of non-empty strings."""
    if raw_models is None:
        return []
    if isinstance(raw_models, str):
        return [raw_models] if raw_models.strip() else []
    if not isinstance(raw_models, Sequence):
        raise ValueError("Embedding model lists must be a sequence of strings.")

    models: list[str] = []
    for value in raw_models:
        if not isinstance(value, str):
            raise ValueError("Embedding model list entries must be strings.")
        if value.strip():
            models.append(value)
    return models


# Private helper: centralizes dataset validation for all public loader paths.
def _load_documents(dataset_name: str) -> Sequence[Mapping[str, Any]]:
    """Load in-memory docs for a dataset with explicit validation."""
    loaded = in_memory_dataset_loader(dataset_name, corpus_only=True)
    if loaded is None:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    objects, _ = loaded
    if not objects:
        raise ValueError(f"Dataset '{dataset_name}' returned zero documents")
    return objects

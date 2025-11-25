"""Database loading and collection management utilities."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import weaviate
import weaviate.collections.classes.config as wvcc

from .database_registry import resolve_spec
from .dataset import in_memory_dataset_loader
from .utils import (
    get_weaviate_client,
    load_config,
    pretty_print_in_memory_document,
    add_tag_to_name,
)


def _drop_and_create_collection(
    client: weaviate.WeaviateClient,
    name: str,
    properties: Sequence[wvcc.Property],
    vector_config: Any,
    recreate: bool = True,
) -> None:
    """Drop (if exists) and create a Weaviate collection."""
    if recreate and client.collections.exists(name):
        client.collections.delete(name)
    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            vector_config=vector_config,
            properties=list(properties),
        )


def _batch_insert(
    client: weaviate.WeaviateClient,
    collection: str,
    items: Sequence[Mapping[str, Any]],
    item_to_props: Callable[[Mapping[str, Any]], Dict[str, Any]],
    batch_size: int = 20,
    verbose: bool = True,
) -> int:
    """
    Insert items into a Weaviate collection in batches.
    
    Returns the total number of items inserted.
    """
    start = time.perf_counter()
    total = 0

    if verbose:
        print(f"Inserting {len(items)} objects into collection '{collection}'...")

    with client.batch.fixed_size(batch_size=batch_size) as batch:
        for i, item in enumerate(items, start=1):
            props = item_to_props(item)
            batch.add_object(collection=collection, properties=props)

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


def get_vector_config(embedding_model: Optional[str] = None) -> Any:
    """
    Factory function to create text2vec_weaviate vectorizer config.
    
    Args:
        embedding_model: Specific model to use. If None, uses default.
    
    Returns:
        Vectorizer configuration object
    """
    if embedding_model:
        return wvcc.Configure.Vectorizer.text2vec_weaviate(model=embedding_model)
    return wvcc.Configure.Vectorizer.text2vec_weaviate()


def create_collection_with_vector_config(
    client: weaviate.WeaviateClient,
    dataset_name: str,
    tag: str = "Default",
    embedding_model: Optional[str] = None,
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
    """
    print(f"Loading dataset '{dataset_name}'...")
    objects, _ = in_memory_dataset_loader(dataset_name)

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
    )

    print(f"Populating collection with {len(objects)} objects...")
    _batch_insert(
        client,
        collection=collection_name,
        items=objects,
        item_to_props=spec.item_to_props,
    )
    print(f"Collection '{collection_name}' ready!\n")


def database_loader(recreate: bool = True, tag: str = "Default") -> None:
    """
    Load dataset from config and populate Weaviate collection.
    
    Args:
        recreate: Whether to drop existing collection before creating
        tag: Suffix to add to collection name
    """
    config_path = Path(os.path.dirname(__file__), "benchmark-config.yml")
    config = load_config(config_path)

    client = get_weaviate_client()

    try:
        dataset_name: str = config["dataset"]
        objects, _ = in_memory_dataset_loader(dataset_name)

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
        )
    finally:
        client.close()
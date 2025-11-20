from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from pydantic import AnyHttpUrl
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple, Optional

import weaviate.collections.classes.config as wvcc
import weaviate

from query_agent_benchmarking.dataset import in_memory_dataset_loader
from query_agent_benchmarking.utils import (
    get_weaviate_client,
    load_config,
    pretty_print_in_memory_document,
    pascalize_name,
    add_tag_to_name,
)

@dataclass()
class DatasetSpec:
    """Defines how to store a dataset in Weaviate."""
    # name_fn should just produce the canonical name (no add_default needed!),
    # e.g. "bright/biology" -> "BrightBiology"
    name_fn: Callable[[str], str]
    properties: Tuple[wvcc.Property, ...]
    vector_config: Any
    item_to_props: Callable[[Mapping[str, Any]], Dict[str, Any]]

TEXT = wvcc.DataType.TEXT
BLOB = wvcc.DataType.BLOB
INT = wvcc.DataType.INT
FIELD = wvcc.Tokenization.FIELD

REGISTRY: list[Tuple[Callable[[str], bool], DatasetSpec]] = [
    # enron
    (
        lambda d: d == "enron",
        DatasetSpec(
            name_fn=lambda d: "EnronEmails",
            properties=(
                wvcc.Property(name="email_body", data_type=TEXT),
                wvcc.Property(
                    name="dataset_id", 
                    data_type=TEXT, 
                    index_searchable=False, 
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "email_body": item["email_body"],
                "dataset_id": str(item["dataset_id"]),
            },
        ),
    ),
    # beir/<subset>
    (
        lambda d: d.startswith("beir/"),
        DatasetSpec(
            name_fn=lambda d: f"Beir{pascalize_name(d.split('beir/')[1])}",
            properties=(
                wvcc.Property(name="title", data_type=TEXT),
                wvcc.Property(name="content", data_type=TEXT),
                wvcc.Property(
                    name="dataset_id", 
                    data_type=TEXT, 
                    index_searchable=False, 
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "title": item["title"],
                "content": item["content"],
                "dataset_id": str(item["doc_id"]),
            },
        ),
    ),
    # bright/<subset>
    (
        lambda d: d.startswith("bright/"),
        DatasetSpec(
            name_fn=lambda d: f"Bright{pascalize_name(d.split('/')[1])}",
            properties=(
                wvcc.Property(name="content", data_type=TEXT),
                wvcc.Property(
                    name="dataset_id", 
                    data_type=TEXT, 
                    index_searchable=False, 
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "content": item["content"],
                "dataset_id": str(item["dataset_id"]),
            },
        ),
    ),
    # lotte/<subset>
    (
        lambda d: d.startswith("lotte/"),
        DatasetSpec(
            name_fn=lambda d: f"Lotte{pascalize_name(d.split('/')[1])}",
            properties=(
                wvcc.Property(name="content", data_type=TEXT),
                wvcc.Property(
                    name="dataset_id", 
                    data_type=TEXT, 
                    index_searchable=False, 
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "content": item["text"],
                "dataset_id": str(item["doc_id"]),
            },
        ),
    ),
    # wixqa
    (
        lambda d: d == "wixqa",
        DatasetSpec(
            name_fn=lambda d: "WixKB",
            properties=(
                wvcc.Property(name="contents", data_type=TEXT),
                wvcc.Property(name="title", data_type=TEXT),
                wvcc.Property(name="article_type", data_type=TEXT, index_searchable=False, index_filterable=False),
                wvcc.Property(
                    name="dataset_id", 
                    data_type=TEXT, 
                    index_searchable=False, 
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "contents": item["contents"],
                "title": item["title"],
                "article_type": item["article_type"],
                "dataset_id": str(item["id"]),
            },
        ),
    ),
    # freshstack-<topic>
    (
        lambda d: d.startswith("freshstack-"),
        DatasetSpec(
            name_fn=lambda d: f"Freshstack{pascalize_name(d.split('-')[1])}",
            properties=(
                wvcc.Property(name="docs_text", data_type=TEXT),
                wvcc.Property(
                    name="dataset_id", 
                    data_type=TEXT, 
                    index_searchable=False, 
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "docs_text": item["text"],
                "dataset_id": str(item["dataset_id"]),
            },
        ),
    ),
    # irpapers/images
    (
        lambda d: d == "irpapers/images",
        DatasetSpec(
            name_fn=lambda d: "IRPapersImages",
            properties=(
                wvcc.Property(name="base64_str", data_type=BLOB),
                wvcc.Property(
                    name="dataset_id",
                    data_type=TEXT,
                    index_searchable=False,
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                )
            ),
            vector_config=wvcc.Configure.MultiVectors.multi2vec_weaviate(
                base_url=AnyHttpUrl("https://dev-embedding.labs.weaviate.io"),
                image_fields=["base64_str"],
                model="ModernVBERT/colmodernvbert",
            ),
            item_to_props=lambda item: {
                "base64_str": item["base64_str"],
                "dataset_id": str(item["dataset_id"]),
            },
        ),
    ),
    # irpapers/text
    (
        lambda d: d == "irpapers/text",
        DatasetSpec(
            name_fn=lambda d: "IRPapersText",
            properties=(
                wvcc.Property(name="content", data_type=TEXT),
                wvcc.Property(
                    name="dataset_id",
                    data_type=TEXT,
                    index_searchable=False,
                    index_filterable=True,
                    skip_vectorization=True,
                    tokenization=FIELD,
                ),
            ),
            vector_config=wvcc.Configure.Vectors.text2vec_weaviate(),
            item_to_props=lambda item: {
                "content": item["transcription"],
                "dataset_id": str(item["dataset_id"]),
            },
        ),
    ),
]

def resolve_spec(dataset_name: str) -> DatasetSpec:
    for pred, spec in REGISTRY:
        if pred(dataset_name):
            return spec
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")

def _drop_and_create_collection(
    weaviate_client: weaviate.WeaviateClient,
    name: str,
    properties: Sequence[wvcc.Property],
    vector_config: Any,
    recreate: bool = True,
) -> None:
    if recreate and weaviate_client.collections.exists(name):
        weaviate_client.collections.delete(name)
    if not weaviate_client.collections.exists(name):
        weaviate_client.collections.create(
            name=name,
            vector_config=vector_config,
            properties=list(properties),
        )

def _batch_insert(
    weaviate_client: weaviate.WeaviateClient,
    collection: str,
    items: Iterable[Mapping[str, Any]],
    item_to_props: Callable[[Mapping[str, Any]], Dict[str, Any]],
    batch_size: int = 20,
):
    start = time.perf_counter()
    total = 0
    print(f"Inserting {len(items)} objects into collection '{collection}'...")
    with weaviate_client.batch.fixed_size(batch_size=batch_size) as batch:
        for i, item in enumerate(items, start=1):
            props = item_to_props(item)
            batch.add_object(collection=collection, properties=props)
            if i % batch_size == 0:
                elapsed = time.perf_counter() - start
                print(f"\033[92mInserted {i} objects ({(elapsed):.1f} s, {(i / max(elapsed, 1e-9)):.1f} objs/s)\033[0m")
            total = i
    elapsed = time.perf_counter() - start
    print(f"Inserted {total} objects in {(elapsed):.2f} s ({(total / max(elapsed, 1e-9)):.1f} objs/s)")

def get_vector_config(embedding_model: Optional[str] = None) -> Any:
    """
    Factory function to create text2vec_weaviate vectorizer config.
    
    Args:
        embedding_model: Specific model to use (e.g., "Snowflake/snowflake-arctic-embed-l-v2.0").
                        If None, uses default model.
    
    Returns:
        Vectorizer configuration object
    """
    if embedding_model:
        return wvcc.Configure.Vectorizer.text2vec_weaviate(
            model=embedding_model
        )
    else:
        # Default config without specifying model
        return wvcc.Configure.Vectorizer.text2vec_weaviate()

def create_collection_with_vector_config(
    weaviate_client: weaviate.WeaviateClient,
    dataset_name: str,
    tag: str = "Default",
    embedding_model: Optional[str] = None,
) -> None:
    """
    Create and populate a collection with a specified embedding model, using a collection name suffixed with a tag.
    
    This is used for embedding model comparison where temporary collections
    are created with different models using the text2vec_weaviate vectorizer.
    
    Args:
        weaviate_client: Connected Weaviate client
        dataset_name: Name of the dataset to load
        tag: Suffix to add to the collection's name (e.g., 'Default', 'Arctic2')
        embedding_model: Embedding model to use (e.g., "Snowflake/snowflake-arctic-embed-l-v2.0").
                        If None, uses default model.
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
        weaviate_client,
        collection_name,
        properties=spec.properties,
        vector_config=vector_config,
        recreate=True,
    )

    print(f"Populating collection with {len(objects)} objects...")
    _batch_insert(
        weaviate_client,
        collection=collection_name,
        items=objects,
        item_to_props=spec.item_to_props,
    )
    print(f"Collection '{collection_name}' ready!\n")


def database_loader(recreate: bool = True, tag: str = "Default") -> None:
    config_path = Path(os.path.dirname(__file__), "benchmark-config.yml")
    config = load_config(config_path)

    weaviate_client = get_weaviate_client()
    
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
            weaviate_client,
            collection_name,
            properties=spec.properties,
            vector_config=spec.vector_config,
            recreate=recreate,
        )

        alias_info = weaviate_client.alias.get(alias_name=alias_collection_name)
        if alias_info is None:
            weaviate_client.alias.create(
                alias_name=alias_collection_name,
                target_collection=collection_name,
            )
        else:
            weaviate_client.alias.update(
                alias_name=alias_collection_name,
                new_target_collection=collection_name,
            )

        _batch_insert(
            weaviate_client,
            collection=collection_name,
            items=objects,
            item_to_props=spec.item_to_props,
        )
    finally:
        weaviate_client.close()
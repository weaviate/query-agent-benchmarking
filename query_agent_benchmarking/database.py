from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import weaviate.collections.classes.config as wvcc
import weaviate

from query_agent_benchmarking.dataset import in_memory_dataset_loader
from query_agent_benchmarking.utils import (
    get_weaviate_client,
    load_config,
    pretty_print_in_memory_document,
)

@dataclass(frozen=True)
class DatasetSpec:
    """Defines how to store a dataset in Weaviate."""
    name_fn: Callable[[str], str]
    properties: Tuple[wvcc.Property, ...]
    vectorizer_config: Any
    item_to_props: Callable[[Mapping[str, Any]], Dict[str, Any]]

def _pascalize_name(raw: str) -> str:
    # Keep letters/digits, split on non-alnum, PascalCase tokens: "beir/scifact" -> "BeirScifact"
    tokens = re.split(r"[^0-9A-Za-z]+", raw)
    return "".join(t.capitalize() for t in tokens if t)

def _drop_and_create_collection(
    weaviate_client: weaviate.WeaviateClient,
    name: str,
    properties: Sequence[wvcc.Property],
    vectorizer_config: Any,
    recreate: bool = True,
) -> None:
    if recreate and weaviate_client.collections.exists(name):
        weaviate_client.collections.delete(name)
    if not weaviate_client.collections.exists(name):
        weaviate_client.collections.create(
            name=name,
            vectorizer_config=vectorizer_config,
            properties=list(properties),
        )

TEXT = wvcc.DataType.TEXT
FIELD = wvcc.Tokenization.FIELD

REGISTRY: List[Tuple[Callable[[str], bool], DatasetSpec]] = [
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
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
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
            name_fn=lambda d: f"Beir{_pascalize_name(d.split('beir/')[1])}",
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
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
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
            name_fn=lambda d: f"Bright{_pascalize_name(d.split('/')[1])}",
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
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
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
            name_fn=lambda d: f"Lotte{_pascalize_name(d.split('/')[1])}",
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
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
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
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
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
            name_fn=lambda d: f"Freshstack{_pascalize_name(d.split('-')[1])}",
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
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            item_to_props=lambda item: {
                "docs_text": item["text"],
                "dataset_id": str(item["dataset_id"]),
            },
        ),
    ),
]

def _resolve_spec(dataset_name: str) -> DatasetSpec:
    for pred, spec in REGISTRY:
        if pred(dataset_name):
            return spec
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")

def _batch_insert(
    weaviate_client: weaviate.WeaviateClient,
    collection: str,
    items: Iterable[Mapping[str, Any]],
    item_to_props: Callable[[Mapping[str, Any]], Dict[str, Any]],
    batch_size: int = 100,
    concurrent_requests: int = 4,
):
    start = time.perf_counter()
    total = 0
    with weaviate_client.batch.fixed_size(batch_size=batch_size, concurrent_requests=concurrent_requests) as batch:
        for i, item in enumerate(items, start=1):
            props = item_to_props(item)
            batch.add_object(collection=collection, properties=props)
            if i % 1000 == 0:
                elapsed = time.perf_counter() - start
                print(f"Inserted {i} objects ({(elapsed):.1f} s, {(i / max(elapsed, 1e-9)):.1f} objs/s)")
            total = i
    elapsed = time.perf_counter() - start
    print(f"Inserted {total} objects in {(elapsed):.2f} s ({(total / max(elapsed, 1e-9)):.1f} objs/s)")

def database_loader(recreate: bool = True) -> None:
    config_path = Path(os.path.dirname(__file__), "benchmark-config.yml")
    config = load_config(config_path)

    weaviate_client = get_weaviate_client()
    
    try:
        dataset_name: str = config["dataset"]
        objects, _ = in_memory_dataset_loader(dataset_name)

        print("\033[92mFirst Document:\033[0m")
        pretty_print_in_memory_document(objects[0]["content"])

        spec = _resolve_spec(dataset_name)
        collection_name = spec.name_fn(dataset_name)

        _drop_and_create_collection(
            weaviate_client,
            collection_name,
            properties=spec.properties,
            vectorizer_config=spec.vectorizer_config,
            recreate=recreate,
        )

        _batch_insert(
            weaviate_client,
            collection=collection_name,
            items=objects,
            item_to_props=spec.item_to_props,
        )
    finally:
        weaviate_client.close()
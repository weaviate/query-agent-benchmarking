"""
Database population service — loads datasets into Weaviate or Engram.

Reads ``database_loader_config.yml`` to determine the dataset and target,
then orchestrates dataset loading, collection creation, and batch insert.
"""

from pathlib import Path

from query_agent_benchmarking.internal.adapters.database.database_config import validate_database_dataset
from query_agent_benchmarking.internal.adapters.database.database_registry import resolve_spec
from query_agent_benchmarking.internal.adapters.database.engram_loader import engram_ingest_all_tenants
from query_agent_benchmarking.internal.adapters.database.database_loader import (
    _drop_and_create_collection,
    _batch_insert,
    _resolve_provider_headers,
    _coerce_model_list,
    _load_documents,
)
from query_agent_benchmarking.internal.adapters.database.naming import add_tag_to_name
from query_agent_benchmarking.internal.adapters.dataset import load_longmemeval_docs_by_tenant
from query_agent_benchmarking.internal.adapters.clients.weaviate_client import get_weaviate_client
from query_agent_benchmarking.internal.config.loader import load_config
from query_agent_benchmarking.internal.core.domain.display import pretty_print_in_memory_document


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "database_loader_config.yml"
)


def populate_db(recreate: bool = True, tag: str = "Default") -> None:
    """
    Load dataset from config and populate the target database.

    Reads ``database_loader_config.yml`` to determine the dataset and target.
    When ``database_target`` is ``"engram"``, sessions are ingested into Engram
    (no Weaviate connection required). Otherwise defaults to Weaviate.

    Args:
        recreate: Whether to drop existing collection before creating.
        tag: Suffix to add to collection name.
    """
    config = load_config(DEFAULT_CONFIG_PATH)

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

    engram_ingest_all_tenants(docs_by_tenant, poll=True)


def _run_weaviate_loader(
    config: dict,
    dataset_name: str,
    recreate: bool = True,
    tag: str = "Default",
) -> None:
    """Load dataset into a Weaviate collection."""
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


def populate_db_multi(
    dataset_names: list[str] | None = None,
    recreate: bool = True,
    tag: str = "Default",
) -> None:
    """
    Populate multiple datasets into Weaviate in one invocation.

    Reads ``dataset_names`` from the parameter or from
    ``database_loader_config.yml``. Each dataset is loaded, its collection
    created (or recreated), and documents batch-inserted.

    Args:
        dataset_names: List of dataset names to populate. Falls back to
            ``dataset_names`` in the config file.
        recreate: Whether to drop existing collections before creating.
        tag: Suffix to add to collection names.
    """
    config = load_config(DEFAULT_CONFIG_PATH)

    datasets = dataset_names or config.get("dataset_names")
    if not datasets:
        raise ValueError(
            "No dataset_names provided. Specify via parameter or "
            "in database_loader_config.yml (dataset_names list)."
        )

    if isinstance(datasets, str):
        datasets = [datasets]

    for dataset_name in datasets:
        validate_database_dataset(dataset_name)

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
        for i, dataset_name in enumerate(datasets, 1):
            print(f"\n{'=' * 60}")
            print(f"\033[92m[{i}/{len(datasets)}] Populating: {dataset_name}\033[0m")
            print(f"{'=' * 60}\n")

            database_target = config.get("database_target", "weaviate")
            if database_target == "engram":
                _run_engram_loader(config, dataset_name)
                continue

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

        print(f"\n\033[92mDone! Populated {len(datasets)} datasets.\033[0m")
    finally:
        client.close()

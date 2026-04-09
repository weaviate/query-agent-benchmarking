"""Dataset loader registry - maps dataset names to the correct loader adapter.

This is the single dispatch point for resolving which loader to call
for a given dataset name.
"""

from typing import Optional

from query_agent_benchmarking.internal.core.models import InMemoryQuery, InMemoryAskQuery


def load_search_dataset(
    dataset_name: str,
    corpus_only: bool = False,
    queries_only: bool = False,
):
    """Load a built-in search dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "beir/scifact/test").
        corpus_only: Return only documents.
        queries_only: Return only queries.

    Returns:
        Tuple of (docs_list, queries_list), filtered by scope flags.
    """
    if corpus_only and queries_only:
        raise ValueError("`corpus_only` and `queries_only` are mutually exclusive.")

    loaded = _dispatch_search_loader(dataset_name)
    if loaded is None:
        return None

    docs, queries = loaded
    if corpus_only:
        return docs, []
    if queries_only:
        return [], queries
    return docs, queries


def _dispatch_search_loader(dataset_name: str):
    """Dispatch to the correct search dataset loader."""
    from query_agent_benchmarking.internal.adapters.dataset import (
        ir_datasets_loader,
        huggingface_loader,
        local_file_loader,
    )

    if dataset_name.startswith("beir/"):
        return ir_datasets_loader.load_beir(dataset_name)
    if dataset_name.startswith("lotte/"):
        return ir_datasets_loader.load_lotte(dataset_name)
    if dataset_name.startswith("bright/"):
        return huggingface_loader.load_bright(dataset_name)
    if dataset_name.startswith("freshstack-"):
        subset = dataset_name.split("-", 1)[1]
        return huggingface_loader.load_freshstack(subset=subset)
    if dataset_name == "enron":
        return huggingface_loader.load_enron()
    if dataset_name == "wixqa":
        return huggingface_loader.load_wixqa()
    if dataset_name in ("irpapers", "irpapers-text-only"):
        return huggingface_loader.load_irpapers()
    if dataset_name == "multihoprag":
        return huggingface_loader.load_multihoprag()
    if dataset_name == "vidore_v3_hr":
        return huggingface_loader.load_vidore()
    if dataset_name == "longmemeval-s":
        return huggingface_loader.load_longmemeval("weaviate/longmemeval-s-cleaned")
    if dataset_name == "longmemeval-m":
        return huggingface_loader.load_longmemeval("weaviate/longmemeval-m-cleaned")
    if dataset_name == "officeqa":
        return local_file_loader.load_officeqa()
    if dataset_name == "reasonir-biology-subset":
        return huggingface_loader.load_reasonir_biology_subset()

    return None


def load_ask_dataset(
    dataset_name: str,
    longmemeval_subset: Optional[dict] = None,
) -> list[InMemoryAskQuery]:
    """Load ask queries from a supported dataset.

    Args:
        dataset_name: Name of the built-in ask dataset.
        longmemeval_subset: Optional dict with ``users_to_test`` key.

    Returns:
        List of InMemoryAskQuery instances.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    from query_agent_benchmarking.internal.adapters.dataset import (
        huggingface_loader,
        local_file_loader,
    )

    if dataset_name == "irpapers":
        return huggingface_loader.load_ask_irpapers()
    if dataset_name == "multihoprag":
        return huggingface_loader.load_ask_multihoprag()
    if dataset_name == "officeqa":
        return local_file_loader.load_ask_officeqa()
    if dataset_name in ("longmemeval-s", "longmemeval-m"):
        hf_path = {
            "longmemeval-s": "weaviate/longmemeval-s-cleaned",
            "longmemeval-m": "weaviate/longmemeval-m-cleaned",
        }[dataset_name]
        users_to_test = (longmemeval_subset or {}).get("users_to_test")
        return huggingface_loader.load_ask_longmemeval(hf_path, users_to_test=users_to_test)

    raise ValueError(
        f"Unknown ask dataset: {dataset_name}. "
        f"Supported ask datasets: irpapers, multihoprag, officeqa, longmemeval-s, longmemeval-m"
    )

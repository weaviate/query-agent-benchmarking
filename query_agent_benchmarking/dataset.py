"""Backward-compatible facade for dataset loading.

All loader implementations now live in adapters/dataset/.
This module delegates to the registry and re-exports public functions
for backward compatibility.
"""

import json
import random
from typing import Optional

from query_agent_benchmarking.internal.core.models import InMemoryQuery, InMemoryAskQuery

# Delegate to the adapter registry
from query_agent_benchmarking.internal.adapters.dataset.registry import (
    load_search_dataset,
    load_ask_dataset,
)

# Re-export Weaviate loaders
from query_agent_benchmarking.internal.adapters.dataset.weaviate_loader import (
    load_queries_from_weaviate_collection,
    load_ask_queries_from_weaviate,
)

# Re-export LongMemEval doc loader
from query_agent_benchmarking.internal.adapters.dataset.huggingface_loader import (
    load_longmemeval_docs_by_tenant,
)


def in_memory_dataset_loader(
    dataset_name: str,
    corpus_only: bool = False,
    queries_only: bool = False,
):
    """Load a built-in dataset into in-memory corpus/query splits.

    Use ``corpus_only=True`` or ``queries_only=True`` to return only one side
    of the split. These flags are mutually exclusive.
    """
    return load_search_dataset(
        dataset_name,
        corpus_only=corpus_only,
        queries_only=queries_only,
    )


def in_memory_ask_dataset_loader(
    dataset_name: str,
    longmemeval_subset: Optional[dict] = None,
) -> list[InMemoryAskQuery]:
    """Load ask queries from a supported dataset.

    Args:
        dataset_name: Name of the built-in ask dataset.
        longmemeval_subset: Optional dict with ``users_to_test`` key containing
            a ``[lower_bound, upper_bound)`` list for tenant filtering.
    """
    return load_ask_dataset(dataset_name, longmemeval_subset=longmemeval_subset)


# ============================================================================
# Utility functions (kept here for backward compatibility)
# ============================================================================

def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)

    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    return train_data, test_data


def _load_dataset_from_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

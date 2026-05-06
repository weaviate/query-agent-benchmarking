"""Build a content-hash index mapping Engram memories back to source sessions.

After ingestion, each memory can be traced to the run (and therefore session)
that created or last updated it.  This module fetches the current content of
every memory touched during ingestion and builds a lookup table keyed by a
SHA-256 hash of the content string.

The index is persisted to ``console/results/`` so the evaluation UI can
annotate retrieved memories with their source session.
"""

import hashlib
import json
import os
from typing import Any, Optional

from query_agent_benchmarking.internal.adapters.database.engram_loader import (
    IngestionResult,
)
from query_agent_benchmarking.internal.adapters.results.serialization import (
    RESULTS_DIR,
    _ensure_results_dir,
)


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of a memory content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_and_save_memory_index(
    result: IngestionResult,
    dataset_name: str,
    manifest_timestamp: str,
    engram_base_url: str = "https://dev-engram.labs.weaviate.io",
    engram_api_key: Optional[str] = None,
    user_id_prefix: str = "longmemeval-",
    group: str = "default",
    verbose: bool = True,
) -> dict:
    """Build a content-hash → source-session index and persist it.

    For every memory ID recorded in ``result.run_completions``, fetches the
    current content from Engram and maps ``hash(content)`` to the session
    that last touched it.

    The index is saved to
    ``console/results/engram-memory-index-{dataset}-{timestamp}.json``
    using the same timestamp as the companion manifest.

    Returns:
        The index dict.
    """
    from engram import EngramClient
    from engram._http import APIError

    completions = result.run_completions or {}
    if not completions:
        return {}

    client = EngramClient(
        api_key=engram_api_key or os.environ["ENGRAM_API_KEY"],
        base_url=engram_base_url,
    )

    # Build memory_id → provenance mapping.
    # A memory may be touched by multiple runs (created by one, updated by
    # another). We record ALL operations so the user can see the full history,
    # but the content hash reflects the final state.
    provenance: dict[str, list[dict[str, str]]] = {}  # memory_id -> [{run_id, session_id, tenant_id, op}]

    for rec in result.run_records:
        comp = completions.get(rec.run_id)
        if not comp:
            continue
        for mid in comp.created_memory_ids:
            provenance.setdefault(mid, []).append({
                "run_id": rec.run_id,
                "session_id": rec.session_id,
                "tenant_id": rec.tenant_id,
                "operation": "created",
            })
        for mid in comp.updated_memory_ids:
            provenance.setdefault(mid, []).append({
                "run_id": rec.run_id,
                "session_id": rec.session_id,
                "tenant_id": rec.tenant_id,
                "operation": "updated",
            })
        for mid in comp.deleted_memory_ids:
            provenance.setdefault(mid, []).append({
                "run_id": rec.run_id,
                "session_id": rec.session_id,
                "tenant_id": rec.tenant_id,
                "operation": "deleted",
            })

    # Fetch current content for each unique memory_id
    if verbose:
        print(f"Building memory index: fetching content for {len(provenance)} memories...")

    index: dict[str, Any] = {}  # content_hash -> entry
    fetched = 0
    not_found = 0

    for memory_id, ops in provenance.items():
        # Determine the tenant for this memory (use the first operation's tenant)
        tenant_id = ops[0]["tenant_id"]
        user_id = f"{user_id_prefix}{tenant_id}"

        content: Optional[str] = None
        try:
            memory = client.memories.get(memory_id, user_id=user_id, group=group)
            content = memory.content
        except APIError as e:
            if e.status_code == 404:
                not_found += 1
            else:
                if verbose:
                    print(f"  Error fetching memory {memory_id}: {e}")
                continue
        except Exception as e:
            if verbose:
                print(f"  Error fetching memory {memory_id}: {e}")
            continue

        fetched += 1

        if content is not None:
            chash = _content_hash(content)
            index[chash] = {
                "memory_id": memory_id,
                "content": content,
                "operations": ops,
            }

        if verbose and fetched % 100 == 0:
            print(f"  Fetched {fetched}/{len(provenance)}")

    if verbose:
        print(f"  Memory index built: {len(index)} entries ({not_found} superseded/deleted)")

    # Persist
    index_doc = {
        "type": "engram_memory_index",
        "dataset": dataset_name,
        "manifest_timestamp": manifest_timestamp,
        "total_memories": len(index),
        "total_superseded": not_found,
        "index": index,
    }

    _ensure_results_dir()
    # Derive timestamp from manifest_timestamp ISO string
    ts = manifest_timestamp.replace(":", "").replace("-", "")[:15]
    filename = f"engram-memory-index-{dataset_name}-{ts}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(index_doc, f, indent=2)

    if verbose:
        print(f"  Saved memory index to {filepath}")

    return index_doc

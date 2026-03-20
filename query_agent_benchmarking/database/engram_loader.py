"""Engram memory ingestion for LongMemEval datasets."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from engram import EngramClient

from ..dataset import in_memory_dataset_loader
from ..utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).parent / "database_loader_config.yml"


def engram_loader(
    dataset_name: Optional[str] = None,
    engram_api_key: Optional[str] = None,
    engram_base_url: Optional[str] = None,
    user_id_prefix: str = "",
) -> None:
    """
    Load a dataset's documents into Engram as memories.

    Each document's session_text is added as a memory, with the tenant_id
    mapped to an Engram user_id (optionally prefixed with user_id_prefix).

    Args:
        dataset_name: Dataset to load (e.g., "longmemeval-s"). Falls back to config file.
        engram_api_key: Engram API key. Falls back to ENGRAM_API_KEY env var.
        engram_base_url: Engram base URL. Falls back to config file or ENGRAM_BASE_URL env var.
        user_id_prefix: Prefix for Engram user IDs (e.g., "longmemeval-").
    """
    config = load_config(DEFAULT_CONFIG_PATH)

    if dataset_name is None:
        dataset_name = config["dataset_name"]

    api_key = engram_api_key or os.getenv("ENGRAM_API_KEY", "")
    base_url = engram_base_url or config.get("engram_base_url") or ""

    if not api_key:
        raise ValueError("Engram API key is required. Set ENGRAM_API_KEY or pass engram_api_key.")
    if not base_url:
        raise ValueError("Engram base URL is required. Set it in database_loader_config.yml or pass engram_base_url.")

    client = EngramClient(api_key=api_key, base_url=base_url)

    print(f"Loading dataset '{dataset_name}' for Engram ingestion...")
    loaded = in_memory_dataset_loader(dataset_name, corpus_only=True)
    if loaded is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    docs, _ = loaded

    print(f"Ingesting {len(docs)} documents into Engram...")
    start = time.perf_counter()

    for i, doc in enumerate(docs, start=1):
        tenant_id = doc.get("tenant_id", "default")
        user_id = f"{user_id_prefix}{tenant_id}"
        session_text = doc.get("session_text", "")

        if not session_text:
            print(f"  Skipping doc {i} (empty session_text)")
            continue

        client.memories.add(session_text, user_id=user_id)

        if i % 50 == 0:
            elapsed = time.perf_counter() - start
            rate = i / max(elapsed, 1e-9)
            print(f"\033[92mIngested {i}/{len(docs)} memories ({elapsed:.1f}s, {rate:.1f}/s)\033[0m")

    elapsed = time.perf_counter() - start
    print(f"Ingested {len(docs)} memories in {elapsed:.2f}s")

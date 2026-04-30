"""Delete all Engram memories for LongMemEval tenants.

Useful for cleaning up after a crashed ingestion run before retrying.
Loads tenant IDs from the same HuggingFace dataset used by populate-db,
then fetches and deletes all memories for each tenant.

Usage:
    uv run python3 scripts/cleanup-engram.py
"""

import os
import time

from dotenv import load_dotenv
from datasets import load_dataset
from engram import EngramClient, RetrievalConfig

load_dotenv()

ENGRAM_BASE_URL = "https://dev-engram.labs.weaviate.io"
USER_ID_PREFIX = "longmemeval-"
GROUP = "default"
DATASET = "weaviate/longmemeval-s-cleaned"
FETCH_LIMIT = 500


def get_all_tenant_ids() -> list[str]:
    """Load all tenant IDs from the HuggingFace dataset."""
    print(f"Loading tenant IDs from {DATASET}...")
    raw_docs = load_dataset(DATASET, "docs")["train"]
    tenant_ids = sorted({str(item["tenant_id"]) for item in raw_docs})
    print(f"  Found {len(tenant_ids)} tenants")
    return tenant_ids


def delete_memories_for_user(client: EngramClient, user_id: str) -> int:
    """Fetch and delete all memories for a user. Returns count deleted."""
    total_deleted = 0

    while True:
        results = client.memories.search(
            query="memory",
            user_id=user_id,
            group=GROUP,
            retrieval_config=RetrievalConfig(
                retrieval_type="hybrid",
                limit=FETCH_LIMIT,
            ),
        )

        if len(results) == 0:
            break

        for memory in results:
            try:
                client.memories.delete(memory.id, user_id=user_id, group=GROUP)
                total_deleted += 1
            except Exception as e:
                print(f"    Failed to delete memory {memory.id}: {e}")

        print(f"    Deleted batch of {len(results)} (total: {total_deleted})")

    return total_deleted


def main():
    client = EngramClient(
        api_key=os.environ["ENGRAM_API_KEY"],
        base_url=ENGRAM_BASE_URL,
    )

    tenant_ids = get_all_tenant_ids()

    total_deleted = 0
    t0 = time.time()

    for i, tenant_id in enumerate(tenant_ids, 1):
        user_id = f"{USER_ID_PREFIX}{tenant_id}"
        print(f"[{i}/{len(tenant_ids)}] Cleaning up {user_id}...")
        deleted = delete_memories_for_user(client, user_id)
        total_deleted += deleted
        if deleted > 0:
            print(f"  Deleted {deleted} memories")
        else:
            print(f"  No memories found")

    elapsed = time.time() - t0
    print(f"\nDone. Deleted {total_deleted} total memories in {elapsed:.0f}s")


if __name__ == "__main__":
    main()

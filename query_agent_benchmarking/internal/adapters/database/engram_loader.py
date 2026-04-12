"""Engram ingestion loader for LongMemEval sessions.

This is the database-population counterpart for Engram, analogous to
``database_loader.py`` for Weaviate collections. It ingests conversation
sessions into Engram's memory system on a per-tenant basis.
"""

import os
import time
from typing import Optional
from dataclasses import dataclass

from engram import EngramClient, ConversationInput, MessageInput


def _parse_session_text(session_text: str) -> ConversationInput:
    """Parse a ``user: ... \\nassistant: ...`` session string into a ConversationInput."""
    messages: list[MessageInput] = []
    current_role: str | None = None
    current_lines: list[str] = []

    for line in session_text.split("\n"):
        if line.startswith("user: ") or line.startswith("assistant: "):
            # Flush the previous message
            if current_role is not None:
                messages.append(
                    MessageInput(role=current_role, content="\n".join(current_lines))
                )
            if line.startswith("user: "):
                current_role = "user"
                current_lines = [line[len("user: "):]]
            else:
                current_role = "assistant"
                current_lines = [line[len("assistant: "):]]
        else:
            # Continuation line of the current message
            current_lines.append(line)

    # Flush the last message
    if current_role is not None:
        messages.append(
            MessageInput(role=current_role, content="\n".join(current_lines))
        )

    return ConversationInput(messages=messages)


@dataclass
class TenantIngestionStats:
    tenant_id: str
    num_sessions: int
    elapsed_seconds: float
    total_created: int = 0
    total_updated: int = 0
    total_deleted: int = 0


def engram_ingest_tenant(
    client: EngramClient,
    tenant_id: str,
    sessions: list[dict],
    group: str = "default",
    user_id_prefix: str = "longmemeval-",
    ingest_delay: float = 0.1,
    poll_interval: float = 2.0,
    verbose: bool = True,
) -> TenantIngestionStats:
    """
    Ingest all sessions for a single tenant into Engram.

    Each session dict must have at least a ``session_text`` key.
    """
    user_id = f"{user_id_prefix}{tenant_id}"
    t0 = time.time()
    run_ids: list[str] = []

    for i, session in enumerate(sessions):
        conversation = _parse_session_text(session["session_text"])
        run = client.memories.add(
            conversation,
            user_id=user_id,
            group=group,
        )
        run_ids.append(run.run_id)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(sessions)}] submitted")

        time.sleep(ingest_delay)

    # Wait for all runs to complete
    if verbose:
        print(f"  Waiting for {len(run_ids)} runs to complete...")
    for i, run_id in enumerate(run_ids):
        while True:
            status = client.runs.get(run_id)
            if status.status in ("completed", "failed", "deleted"):
                break
            time.sleep(poll_interval)
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(run_ids)}] runs completed")

    elapsed = time.time() - t0

    # Collect memory operation counts
    total_created = 0
    total_updated = 0
    total_deleted = 0
    for run_id in run_ids:
        run_status = client.runs.get(run_id)
        ops = run_status.committed_operations
        total_created += len(getattr(ops, "created", []))
        total_updated += len(getattr(ops, "updated", []))
        total_deleted += len(getattr(ops, "deleted", []))

    stats = TenantIngestionStats(
        tenant_id=tenant_id,
        num_sessions=len(sessions),
        elapsed_seconds=elapsed,
        total_created=total_created,
        total_updated=total_updated,
        total_deleted=total_deleted,
    )

    if verbose:
        print(f"  Tenant {tenant_id}: {len(sessions)} sessions in {elapsed:.0f}s")
        print(f"  Created: {total_created}  Updated: {total_updated}  Deleted: {total_deleted}")

    return stats


def engram_ingest_all_tenants(
    docs_by_tenant: dict[str, list[dict]],
    engram_base_url: str = "https://dev-engram.labs.weaviate.io",
    engram_api_key: Optional[str] = None,
    group: str = "default",
    user_id_prefix: str = "longmemeval-",
    ingest_delay: float = 0.1,
    poll_interval: float = 2.0,
    verbose: bool = True,
) -> list[TenantIngestionStats]:
    """
    Ingest sessions for all tenants into Engram.

    Args:
        docs_by_tenant: Dict mapping tenant_id -> list of session dicts.
        engram_base_url: Engram API base URL.
        engram_api_key: Engram API key. Falls back to ``ENGRAM_API_KEY`` env var.
        group: Engram memory group name.
        user_id_prefix: Prefix for Engram user IDs.
        ingest_delay: Seconds to sleep between session submissions.
        poll_interval: Seconds between run-status polls.
        verbose: Print progress updates.

    Returns:
        Per-tenant ingestion stats.
    """
    client = EngramClient(
        api_key=engram_api_key or os.environ["ENGRAM_API_KEY"],
        base_url=engram_base_url,
    )

    all_stats: list[TenantIngestionStats] = []
    tenant_ids = sorted(docs_by_tenant.keys())

    for i, tenant_id in enumerate(tenant_ids):
        if verbose:
            print(f"\n[{i + 1}/{len(tenant_ids)}] Ingesting tenant {tenant_id}")
        stats = engram_ingest_tenant(
            client=client,
            tenant_id=tenant_id,
            sessions=docs_by_tenant[tenant_id],
            group=group,
            user_id_prefix=user_id_prefix,
            ingest_delay=ingest_delay,
            poll_interval=poll_interval,
            verbose=verbose,
        )
        all_stats.append(stats)

    return all_stats

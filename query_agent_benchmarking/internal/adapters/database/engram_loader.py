"""Engram ingestion loader for LongMemEval sessions.

This is the database-population counterpart for Engram, analogous to
``database_loader.py`` for Weaviate collections. It ingests conversation
sessions into Engram's memory system on a per-tenant basis.
"""

import os
import time
from typing import Optional
from dataclasses import dataclass, field

from engram import EngramClient, ConversationInput, MessageInput


def _parse_session_text(session_text: str) -> ConversationInput:
    """Parse a ``user: ... \\nassistant: ...`` session string into a ConversationInput.

    Filters out any messages with empty content, which can occur when the
    dataset contains lines like ``user: \\nassistant: ...`` (empty user turn).
    """
    messages: list[MessageInput] = []
    current_role: str | None = None
    current_lines: list[str] = []

    for line in session_text.split("\n"):
        if line.startswith("user: ") or line.startswith("assistant: "):
            # Flush the previous message
            if current_role is not None:
                content = "\n".join(current_lines).strip()
                if content:
                    messages.append(
                        MessageInput(role=current_role, content=content)
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
        content = "\n".join(current_lines).strip()
        if content:
            messages.append(
                MessageInput(role=current_role, content=content)
            )

    return ConversationInput(messages=messages)


@dataclass
class RunRecord:
    """Tracks a single submitted run for later polling."""
    run_id: str
    tenant_id: str
    submitted_at: float


@dataclass
class IngestionResult:
    """Result of a bulk ingestion run."""
    run_records: list[RunRecord]
    tenant_session_counts: dict[str, int]
    submit_elapsed_seconds: float
    stats: Optional[list["TenantIngestionStats"]] = None


@dataclass
class TenantIngestionStats:
    tenant_id: str
    num_sessions: int
    elapsed_seconds: float
    total_created: int = 0
    total_updated: int = 0
    total_deleted: int = 0
    run_durations: list[float] = field(default_factory=list)


def _submit_all(
    client: EngramClient,
    docs_by_tenant: dict[str, list[dict]],
    group: str,
    user_id_prefix: str,
    ingest_delay: float,
    verbose: bool,
) -> list[RunRecord]:
    """Submit every session across all tenants. Returns run records for polling."""
    records: list[RunRecord] = []
    total = sum(len(sessions) for sessions in docs_by_tenant.values())
    submitted = 0
    skipped = 0

    for tenant_id in sorted(docs_by_tenant.keys()):
        user_id = f"{user_id_prefix}{tenant_id}"
        sessions = docs_by_tenant[tenant_id]

        for session in sessions:
            conversation = _parse_session_text(session["session_text"])
            if not conversation.messages:
                skipped += 1
                if verbose:
                    sid = session.get("session_id", "?")
                    print(f"  Skipped session {sid} for tenant {tenant_id}: no valid messages after parsing")
                continue
            t_submit = time.time()
            try:
                run = client.memories.add(
                    conversation,
                    user_id=user_id,
                    group=group,
                    conversation_id=session.get("session_id"),
                )
            except Exception as e:
                skipped += 1
                if verbose:
                    sid = session.get("session_id", "?")
                    print(f"  Skipped session {sid} for tenant {tenant_id}: {e}")
                continue
            records.append(RunRecord(
                run_id=run.run_id,
                tenant_id=tenant_id,
                submitted_at=t_submit,
            ))
            submitted += 1

            if verbose and submitted % 50 == 0:
                print(f"  Submitted {submitted}/{total}")

            if ingest_delay > 0:
                time.sleep(ingest_delay)

    if verbose:
        print(f"  All {submitted} sessions submitted across {len(docs_by_tenant)} tenants")
        if skipped:
            print(f"  Skipped {skipped} sessions due to errors")

    return records


def _poll_and_collect(
    client: EngramClient,
    records: list[RunRecord],
    poll_interval: float,
    verbose: bool,
) -> dict[str, TenantIngestionStats]:
    """Poll all runs to completion and build per-tenant stats."""
    # Group records by tenant for stats aggregation
    tenant_sessions: dict[str, int] = {}
    for rec in records:
        tenant_sessions[rec.tenant_id] = tenant_sessions.get(rec.tenant_id, 0) + 1

    stats_map: dict[str, TenantIngestionStats] = {}
    completed = 0
    t_poll_start = time.time()

    for rec in records:
        # Poll until done
        try:
            while True:
                status = client.runs.get(rec.run_id)
                if status.status in ("completed", "failed", "deleted"):
                    break
                time.sleep(poll_interval)
        except Exception as e:
            if verbose:
                print(f"  Failed to poll run {rec.run_id} (tenant {rec.tenant_id}): {e}")
            completed += 1
            continue

        completed += 1
        run_duration = time.time() - rec.submitted_at

        # Collect memory operation counts
        ops = status.committed_operations
        created = len(getattr(ops, "created", []) or [])
        updated = len(getattr(ops, "updated", []) or [])
        deleted = len(getattr(ops, "deleted", []) or [])

        if rec.tenant_id not in stats_map:
            stats_map[rec.tenant_id] = TenantIngestionStats(
                tenant_id=rec.tenant_id,
                num_sessions=tenant_sessions[rec.tenant_id],
                elapsed_seconds=0.0,
            )

        tenant_stats = stats_map[rec.tenant_id]
        tenant_stats.total_created += created
        tenant_stats.total_updated += updated
        tenant_stats.total_deleted += deleted
        tenant_stats.run_durations.append(run_duration)

        if verbose and completed % 50 == 0:
            print(f"  Completed {completed}/{len(records)}")

    # Set elapsed_seconds to wall-clock time from first submission to last completion
    for tenant_id, ts in stats_map.items():
        tenant_records = [r for r in records if r.tenant_id == tenant_id]
        first_submit = min(r.submitted_at for r in tenant_records)
        ts.elapsed_seconds = time.time() - first_submit

    if verbose:
        poll_elapsed = time.time() - t_poll_start
        print(f"  All {completed} runs completed (poll phase: {poll_elapsed:.0f}s)")

    return stats_map


def engram_ingest_all_tenants(
    docs_by_tenant: dict[str, list[dict]],
    engram_base_url: str = "https://dev-engram.labs.weaviate.io",
    engram_api_key: Optional[str] = None,
    group: str = "default",
    user_id_prefix: str = "longmemeval-",
    ingest_delay: float = 0.1,
    poll: bool = False,
    poll_interval: float = 2.0,
    verbose: bool = True,
) -> IngestionResult:
    """
    Ingest sessions for all tenants into Engram.

    Submits all sessions across every tenant first. If ``poll=True``, also
    polls every run to completion and populates per-tenant stats with
    operation counts and run durations. Otherwise returns immediately after
    submission (fire-and-forget).

    Args:
        docs_by_tenant: Dict mapping tenant_id -> list of session dicts.
        engram_base_url: Engram API base URL.
        engram_api_key: Engram API key. Falls back to ``ENGRAM_API_KEY`` env var.
        group: Engram memory group name.
        user_id_prefix: Prefix for Engram user IDs.
        ingest_delay: Seconds to sleep between session submissions.
        poll: Whether to poll runs to completion and collect stats.
        poll_interval: Seconds between run-status polls (only used when poll=True).
        verbose: Print progress updates.

    Returns:
        An ``IngestionResult`` with run records and submission timing.
        If ``poll=True``, ``result.stats`` contains per-tenant stats.
    """
    client = EngramClient(
        api_key=engram_api_key or os.environ["ENGRAM_API_KEY"],
        base_url=engram_base_url,
    )

    t0 = time.time()

    # Phase 1: submit everything
    if verbose:
        total = sum(len(s) for s in docs_by_tenant.values())
        print(f"Submitting {total} sessions across {len(docs_by_tenant)} tenants...")
    records = _submit_all(client, docs_by_tenant, group, user_id_prefix, ingest_delay, verbose)

    submit_elapsed = time.time() - t0
    tenant_session_counts = {}
    for rec in records:
        tenant_session_counts[rec.tenant_id] = tenant_session_counts.get(rec.tenant_id, 0) + 1

    result = IngestionResult(
        run_records=records,
        tenant_session_counts=tenant_session_counts,
        submit_elapsed_seconds=submit_elapsed,
    )

    if verbose:
        print(f"  Submission complete in {submit_elapsed:.0f}s")

    if not poll:
        if verbose:
            print("  Polling disabled — returning immediately (fire-and-forget)")
        return result

    # Phase 2: poll all runs to completion
    if verbose:
        print(f"Polling {len(records)} runs for completion...")
    stats_map = _poll_and_collect(client, records, poll_interval, verbose)

    result.stats = [stats_map[tid] for tid in sorted(stats_map)]

    if verbose:
        wall = time.time() - t0
        print(f"\nTotal wall-clock time: {wall:.0f}s")
        for tid in sorted(stats_map):
            s = stats_map[tid]
            avg_dur = sum(s.run_durations) / len(s.run_durations) if s.run_durations else 0
            print(
                f"  {tid}: {s.num_sessions} sessions, "
                f"created={s.total_created} updated={s.total_updated} deleted={s.total_deleted}, "
                f"avg run duration={avg_dur:.1f}s"
            )

    return result

"""Engram ingestion loader for LongMemEval sessions.

This is the database-population counterpart for Engram, analogous to
``database_loader.py`` for Weaviate collections. It ingests conversation
sessions into Engram's memory system on a per-tenant basis.
"""

import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional

from engram import ConversationInput, EngramClient, MessageInput

PRINT_INTERVAL = 10


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
    session_id: str = ""
    session_date: str = ""


@dataclass
class RunCompletionInfo:
    """Per-run completion data collected during polling."""
    status: str
    created: int
    updated: int
    deleted: int
    run_duration_seconds: float
    created_memory_ids: list[str] = field(default_factory=list)
    updated_memory_ids: list[str] = field(default_factory=list)
    deleted_memory_ids: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Result of a bulk ingestion run."""
    run_records: list[RunRecord]
    tenant_session_counts: dict[str, int]
    submit_elapsed_seconds: float
    stats: Optional[list["TenantIngestionStats"]] = None
    run_completions: Optional[dict[str, "RunCompletionInfo"]] = None


@dataclass
class TenantIngestionStats:
    tenant_id: str
    num_sessions: int
    elapsed_seconds: float
    total_created: int = 0
    total_updated: int = 0
    total_deleted: int = 0
    run_durations: list[float] = field(default_factory=list)


def _get_inputs_from_conversation(
    conversation: ConversationInput,
    ingestion_mode: str,
) -> list[ConversationInput]:
    """Split a parsed conversation into the list of inputs to submit to Engram.

    - ``conversation``: one input — the full conversation.
    - ``user_messages``: one input per user message (assistant turns discarded).
    - ``message_turn``: one input per assistant message encountered, accumulating
      all messages since the last flush; any trailing messages after the final
      assistant turn become one last input.
    """
    if ingestion_mode == "conversation":
        return [conversation]

    if ingestion_mode == "user_messages":
        return [
            ConversationInput(messages=[m])
            for m in conversation.messages
            if m.role == "user"
        ]

    if ingestion_mode == "message_turn":
        turns: list[ConversationInput] = []
        current: list[MessageInput] = []
        for msg in conversation.messages:
            current.append(msg)
            if msg.role == "assistant":
                turns.append(ConversationInput(messages=current))
                current = []
        if current:
            turns.append(ConversationInput(messages=current))
        return turns

    raise ValueError(
        f"Unsupported ingestion_mode '{ingestion_mode}'. "
        f"Supported: conversation, user_messages, message_turn"
    )


def _submit_all(
    client: EngramClient,
    docs_by_tenant: dict[str, list[dict]],
    group: str,
    user_id_prefix: str,
    ingest_delay: float,
    verbose: bool,
    ingestion_mode: str = "conversation",
    on_progress: Optional[Callable[[dict], None]] = None,
) -> list[RunRecord]:
    """Submit every session across all tenants. Returns run records for polling.

    Args:
        ingestion_mode: How to split each session into Engram ``add()`` calls.
            ``"conversation"`` submits the full parsed conversation per session.
            ``"user_messages"`` submits each user message individually.
            ``"message_turn"`` submits each user/assistant exchange as one input.
    """
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

            inputs = _get_inputs_from_conversation(conversation, ingestion_mode)
            if not inputs:
                skipped += 1
                if verbose:
                    sid = session.get("session_id", "?")
                    print(f"  Skipped session {sid} for tenant {tenant_id}: no inputs after splitting")
                continue

            for input_idx, conv_input in enumerate(inputs):
                t_submit = time.time()
                try:
                    run = client.memories.add(
                        conv_input,
                        user_id=user_id,
                        group=group,
                        properties={"conversation_id": session.get("session_id")},
                    )
                except Exception as e:
                    skipped += 1
                    if verbose:
                        sid = session.get("session_id", "?")
                        print(f"  Skipped session {sid} input {input_idx} for tenant {tenant_id}: {e}")
                    continue
                records.append(RunRecord(
                    run_id=run.run_id,
                    tenant_id=tenant_id,
                    submitted_at=t_submit,
                    session_id=session.get("session_id", ""),
                    session_date=session.get("session_date", ""),
                ))
                submitted += 1

                if on_progress:
                    on_progress({
                        "phase": "submit",
                        "submitted": submitted,
                        "skipped": skipped,
                        "total": total,
                        "tenant_id": tenant_id,
                        "session_id": session.get("session_id", ""),
                        "input_index": input_idx,
                    })

                if verbose and submitted % PRINT_INTERVAL == 0:
                    print(f"  Submitted {submitted}")

                if ingest_delay > 0:
                    time.sleep(ingest_delay)

    if verbose:
        print(f"  All {submitted} inputs submitted across {len(docs_by_tenant)} tenants (mode: {ingestion_mode})")
        if skipped:
            print(f"  Skipped {skipped} due to errors or empty splits")

    return records


def _poll_and_collect(
    client: EngramClient,
    records: list[RunRecord],
    poll_interval: float,
    verbose: bool,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> tuple[dict[str, TenantIngestionStats], dict[str, RunCompletionInfo]]:
    """Poll all runs to completion and build per-tenant stats + per-run completions."""
    # Group records by tenant for stats aggregation
    tenant_sessions: dict[str, int] = {}
    for rec in records:
        tenant_sessions[rec.tenant_id] = tenant_sessions.get(rec.tenant_id, 0) + 1

    stats_map: dict[str, TenantIngestionStats] = {}
    completions: dict[str, RunCompletionInfo] = {}
    completed = 0
    total_created = 0
    total_updated = 0
    total_deleted = 0
    t_poll_start = time.time()

    for rec in records:
        # Poll until done
        poll_checks = 0
        try:
            while True:
                status = client.runs.get(rec.run_id)
                if status.status in ("completed", "failed", "deleted"):
                    break
                poll_checks += 1
                if on_progress:
                    on_progress({
                        "phase": "poll_wait",
                        "completed": completed,
                        "total": len(records),
                        "tenant_id": rec.tenant_id,
                        "run_id": rec.run_id,
                        "run_status": status.status,
                        "poll_checks": poll_checks,
                        "memories_created": total_created,
                        "memories_updated": total_updated,
                        "memories_deleted": total_deleted,
                    })
                time.sleep(poll_interval)
        except Exception as e:
            if verbose:
                print(f"  Failed to poll run {rec.run_id} (tenant {rec.tenant_id}): {e}")
            completed += 1
            continue

        completed += 1
        run_duration = time.time() - rec.submitted_at

        # Collect memory operation counts and IDs
        ops = status.committed_operations
        created_ops = getattr(ops, "created", []) or []
        updated_ops = getattr(ops, "updated", []) or []
        deleted_ops = getattr(ops, "deleted", []) or []
        created = len(created_ops)
        updated = len(updated_ops)
        deleted = len(deleted_ops)
        total_created += created
        total_updated += updated
        total_deleted += deleted

        completions[rec.run_id] = RunCompletionInfo(
            status=status.status,
            created=created,
            updated=updated,
            deleted=deleted,
            run_duration_seconds=run_duration,
            created_memory_ids=[op.memory_id for op in created_ops],
            updated_memory_ids=[op.memory_id for op in updated_ops],
            deleted_memory_ids=[op.memory_id for op in deleted_ops],
        )

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

        if on_progress:
            on_progress({
                "phase": "poll",
                "completed": completed,
                "total": len(records),
                "tenant_id": rec.tenant_id,
                "run_status": status.status,
                "memories_created": total_created,
            })

        if verbose and completed % PRINT_INTERVAL == 0:
            print(f"  Completed {completed}/{len(records)}")

    # Set elapsed_seconds to wall-clock time from first submission to last completion
    for tenant_id, ts in stats_map.items():
        tenant_records = [r for r in records if r.tenant_id == tenant_id]
        first_submit = min(r.submitted_at for r in tenant_records)
        ts.elapsed_seconds = time.time() - first_submit

    if verbose:
        poll_elapsed = time.time() - t_poll_start
        print(f"  All {completed} runs completed (poll phase: {poll_elapsed:.0f}s)")

    return stats_map, completions


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
    ingestion_mode: str = "conversation",
    on_progress: Optional[Callable[[dict], None]] = None,
    dry_run: bool = False,
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
        ingestion_mode: How to split each session into Engram ``add()`` calls.
            ``"conversation"`` (default) submits the full conversation per session.
            ``"user_messages"`` submits each user message individually.
            ``"message_turn"`` submits each user/assistant exchange as one input,
            accumulating until each assistant message; trailing user-only messages
            become a final input.
        dry_run: If True, count requests without submitting to Engram.
            No API key is required. Returns an ``IngestionResult`` with
            synthetic run records reflecting the would-be request count.

    Returns:
        An ``IngestionResult`` with run records and submission timing.
        If ``poll=True``, ``result.stats`` contains per-tenant stats.
    """
    if dry_run:
        client = _DryRunEngramClient()
        ingest_delay = 0.0
        poll = False
    else:
        client = EngramClient(
            api_key=engram_api_key or os.environ["ENGRAM_API_KEY"],
            base_url=engram_base_url,
        )

    t0 = time.time()

    # Phase 1: submit everything
    if verbose:
        total = sum(len(s) for s in docs_by_tenant.values())
        mode_label = f" (mode: {ingestion_mode})" if ingestion_mode != "conversation" else " (mode: conversation)"
        print(f"Submitting {total} sessions across {len(docs_by_tenant)} tenants{mode_label}...")
    records = _submit_all(client, docs_by_tenant, group, user_id_prefix, ingest_delay, verbose, ingestion_mode, on_progress)

    submit_elapsed = time.time() - t0
    tenant_session_counts = {}
    for rec in records:
        tenant_session_counts[rec.tenant_id] = tenant_session_counts.get(rec.tenant_id, 0) + 1

    if dry_run and verbose:
        print(
            f"\n[Dry run] Would submit {len(records)} requests across {len(tenant_session_counts)} tenants"
        )
        for tid in sorted(tenant_session_counts):
            print(f"  {tid}: {tenant_session_counts[tid]} requests")
        print("[Dry run] No data was sent to Engram.")

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
    stats_map, completions = _poll_and_collect(client, records, poll_interval, verbose, on_progress)

    result.stats = [stats_map[tid] for tid in sorted(stats_map)]
    result.run_completions = completions

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


class _DryRunEngramClient:
    """No-op Engram client for dry-run counting — no HTTP calls made."""

    def __init__(self):
        self._count = 0
        self.memories = self
        self.runs = self

    def add(self, *args, **kwargs):
        self._count += 1
        return SimpleNamespace(run_id=f"dry-run-{self._count}")

    def get(self, *args, **kwargs):
        return SimpleNamespace(
            status="completed",
            committed_operations=SimpleNamespace(created=[], updated=[], deleted=[]),
        )

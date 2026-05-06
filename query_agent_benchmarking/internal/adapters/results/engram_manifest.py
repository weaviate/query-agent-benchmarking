"""Serialize Engram ingestion results to a local manifest file.

After populating Engram, a manifest JSON is written to ``console/results/``
so that run IDs can be traced back to the sessions that produced them.
"""

import json
from datetime import datetime, timezone

from query_agent_benchmarking.internal.adapters.database.engram_loader import (
    IngestionResult,
)
from query_agent_benchmarking.internal.adapters.results.serialization import (
    RESULTS_DIR,
    _ensure_results_dir,
)


def save_engram_manifest(result: IngestionResult, dataset_name: str) -> dict:
    """Serialize an ``IngestionResult`` to a manifest dict and persist it.

    The manifest is written to ``console/results/engram-ingest-{dataset}-{timestamp}.json``.

    Returns:
        The manifest dict (also suitable for returning in an API response).
    """
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d-%H%M%S")
    completions = result.run_completions or {}

    runs = []
    for rec in result.run_records:
        entry: dict = {
            "run_id": rec.run_id,
            "tenant_id": rec.tenant_id,
            "session_id": rec.session_id,
            "session_date": rec.session_date,
            "submitted_at": rec.submitted_at,
        }
        comp = completions.get(rec.run_id)
        if comp:
            entry["status"] = comp.status
            entry["created"] = comp.created
            entry["updated"] = comp.updated
            entry["deleted"] = comp.deleted
            entry["run_duration_seconds"] = round(comp.run_duration_seconds, 2)
        runs.append(entry)

    stats_list = None
    if result.stats:
        stats_list = [
            {
                "tenant_id": s.tenant_id,
                "num_sessions": s.num_sessions,
                "elapsed_seconds": round(s.elapsed_seconds, 2),
                "total_created": s.total_created,
                "total_updated": s.total_updated,
                "total_deleted": s.total_deleted,
            }
            for s in result.stats
        ]

    manifest: dict = {
        "type": "engram_ingest",
        "timestamp": now.isoformat(),
        "dataset": dataset_name,
        "submit_elapsed_seconds": round(result.submit_elapsed_seconds, 2),
        "total_sessions": len(result.run_records),
        "total_tenants": len(result.tenant_session_counts),
        "tenant_session_counts": result.tenant_session_counts,
        "runs": runs,
    }
    if stats_list is not None:
        manifest["stats"] = stats_list

    _ensure_results_dir()
    filename = f"engram-ingest-{dataset_name}-{timestamp_str}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

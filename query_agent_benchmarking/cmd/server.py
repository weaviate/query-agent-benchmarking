"""
THIS IS A CONCEPTUAL PLACEHOLDER. IT IS NOT REALLY USED FOR ANYTHING!

Toy benchmark server — run benchmarks via HTTP.

Start:
    uv run uvicorn query_agent_benchmarking.cmd.server:app --reload

Endpoints:
    POST /search    — run a search benchmark
    POST /ask       — run an ask benchmark
    POST /compare   — compare embedding models
    POST /populate  — populate the database
    GET  /health    — health check
"""

import json
import os
import queue
import traceback
import threading
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import query_agent_benchmarking

app = FastAPI(
    title="Query Agent Benchmarking",
    description="Run Weaviate Query Agent benchmarks via HTTP",
)


# ---------------------------------------------------------------------------
# Job tracking for long-running Engram ingestion
# ---------------------------------------------------------------------------
# Stores progress and results so clients can reconnect after navigating away.

_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RunSearchBenchmarkRequest(BaseModel):
    search_dataset: Optional[str] = None
    agent_name: Optional[str] = None
    num_trials: Optional[int] = None
    use_subset: Optional[bool] = None
    num_samples: Optional[int] = None
    use_async: Optional[bool] = None
    batch_size: Optional[int] = None
    max_concurrent: Optional[int] = None
    embedding_model: Optional[str] = None
    text_embedding_model: Optional[str] = None
    search_target: Optional[str] = None


class RunAskBenchmarkRequest(BaseModel):
    queries: Optional[str] = None  # built-in dataset name
    ask_dataset: Optional[str] = None
    agent_name: Optional[str] = None
    judge_model: Optional[str] = None
    ensemble_k: Optional[int] = None
    use_reasoning: Optional[bool] = None
    num_trials: Optional[int] = None
    use_subset: Optional[bool] = None
    num_samples: Optional[int] = None
    use_async: Optional[bool] = None
    batch_size: Optional[int] = None
    max_concurrent: Optional[int] = None
    embedding_model: Optional[str] = None
    longmemeval_subset_start: Optional[int] = None
    longmemeval_subset_end: Optional[int] = None
    longmemeval_tenant_ids: Optional[list[str]] = None


class CompareEmbeddingsRequest(BaseModel):
    search_dataset: Optional[str] = None
    agent_names: Optional[list[str]] = None
    embedding_models: Optional[list[str]] = None
    num_trials: Optional[int] = None
    use_subset: Optional[bool] = None
    num_samples: Optional[int] = None


class PopulateDatabaseRequest(BaseModel):
    recreate: bool = True
    tag: str = "Default"
    dataset_name: Optional[str] = None
    database_target: Optional[str] = None
    text_embedding_model: Optional[str] = None
    image_embedding_model: Optional[str] = None
    use_MUVERA_encoding: Optional[bool] = None
    ksim: Optional[int] = None
    dprojections: Optional[int] = None
    repetitions: Optional[int] = None
    ef: Optional[int] = None
    longmemeval_subset_start: Optional[int] = None
    longmemeval_subset_end: Optional[int] = None
    longmemeval_tenant_ids: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-search-benchmark")
def run_search(req: RunSearchBenchmarkRequest) -> dict[str, Any]:
    """Run a search benchmark with the given parameters."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = query_agent_benchmarking.run_search_eval(**kwargs)
        return {"status": "ok", "result": result}
    except Exception:
        traceback.print_exc()
        return {"status": "error", "error": "An internal error occurred."}


@app.post("/run-ask-benchmark")
def run_ask(req: RunAskBenchmarkRequest) -> dict[str, Any]:
    """Run an ask benchmark with the given parameters."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        # Convert flat subset fields to the nested dict expected by run_ask_eval
        tenant_ids = kwargs.pop("longmemeval_tenant_ids", None)
        start = kwargs.pop("longmemeval_subset_start", None)
        end = kwargs.pop("longmemeval_subset_end", None)
        if tenant_ids:
            kwargs["longmemeval_subset"] = {"tenant_ids": tenant_ids}
        elif start is not None and end is not None:
            kwargs["longmemeval_subset"] = {"users_to_test": [start, end]}
        result = query_agent_benchmarking.run_ask_eval(**kwargs)
        return {"status": "ok", "result": result}
    except Exception:
        traceback.print_exc()
        return {"status": "error", "error": "An internal error occurred."}


@app.post("/compare-embeddings")
def run_compare(req: CompareEmbeddingsRequest) -> dict[str, Any]:
    """Compare embedding models."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = query_agent_benchmarking.compare_embeddings(**kwargs)
        return {"status": "ok", "result": result}
    except Exception:
        traceback.print_exc()
        return {"status": "error", "error": "An internal error occurred."}


def _build_populate_config(req: PopulateDatabaseRequest) -> dict:
    """Load base config and apply request overrides."""
    from pathlib import Path
    from query_agent_benchmarking.internal.config.loader import load_config

    config_path = (
        Path(__file__).resolve().parent.parent
        / "internal" / "config" / "database_loader_config.yml"
    )
    config = load_config(str(config_path))

    if req.dataset_name is not None:
        config["dataset_name"] = req.dataset_name
    if req.database_target is not None:
        config["database_target"] = req.database_target
    if req.text_embedding_model is not None:
        config["text_embedding_model"] = req.text_embedding_model
    if req.image_embedding_model is not None:
        config["image_embedding_model"] = req.image_embedding_model
    if req.use_MUVERA_encoding is not None:
        config["use_MUVERA_encoding"] = req.use_MUVERA_encoding
    if req.ksim is not None:
        config["ksim"] = req.ksim
    if req.dprojections is not None:
        config["dprojections"] = req.dprojections
    if req.repetitions is not None:
        config["repetitions"] = req.repetitions
    if req.ef is not None:
        config["ef"] = req.ef
    if req.longmemeval_tenant_ids:
        config["longmemeval_subset"] = {
            "tenant_ids": req.longmemeval_tenant_ids,
        }
    elif req.longmemeval_subset_start is not None and req.longmemeval_subset_end is not None:
        config["longmemeval_subset"] = {
            "users_to_test": [req.longmemeval_subset_start, req.longmemeval_subset_end]
        }

    return config


@app.post("/populate-db")
def populate_db_endpoint(req: PopulateDatabaseRequest) -> dict[str, Any]:
    """Populate the database from config, with optional overrides."""
    from query_agent_benchmarking.internal.adapters.database.database_config import validate_database_dataset
    from query_agent_benchmarking.internal.core.services.populate_db import (
        _run_engram_loader,
        _run_weaviate_loader,
    )

    try:
        config = _build_populate_config(req)
        dataset_name = config["dataset_name"]
        validate_database_dataset(dataset_name)
        database_target = config.get("database_target", "weaviate")

        if database_target == "engram":
            manifest = _run_engram_loader(config, dataset_name)
            return {"status": "ok", "engram_manifest": manifest}
        else:
            _run_weaviate_loader(config, dataset_name, recreate=req.recreate, tag=req.tag)
            return {"status": "ok"}
    except Exception:
        traceback.print_exc()
        return {"status": "error", "error": "An internal error occurred."}


@app.post("/populate-db-stream")
def populate_db_stream(req: PopulateDatabaseRequest):
    """Populate Engram with SSE progress streaming.

    Sends newline-delimited JSON events:
      {"event": "started", "job_id": "..."}
      {"event": "loading", "message": "..."}
      {"event": "progress", "phase": "submit", "submitted": 5, "total": 120, ...}
      {"event": "progress", "phase": "poll", "completed": 3, "total": 120, ...}
      {"event": "complete", "manifest": {...}}
      {"event": "error", "message": "..."}

    Job state is also tracked in memory so clients can reconnect via
    ``GET /populate-db-job/{job_id}`` after navigating away.
    """
    from query_agent_benchmarking.internal.adapters.database.database_config import validate_database_dataset
    from query_agent_benchmarking.internal.adapters.database.engram_loader import engram_ingest_all_tenants
    from query_agent_benchmarking.internal.adapters.dataset import load_longmemeval_docs_by_tenant
    from query_agent_benchmarking.internal.adapters.results.engram_manifest import save_engram_manifest

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": None, "manifest": None, "error": None}

    progress_queue: queue.Queue[dict | None] = queue.Queue()

    def on_progress(event: dict) -> None:
        # Update the shared job state so poll clients can see it
        _jobs[job_id]["progress"] = event
        progress_queue.put(event)

    def run_ingestion() -> None:
        try:
            config = _build_populate_config(req)
            dataset_name = config["dataset_name"]
            validate_database_dataset(dataset_name)

            loading_event = {"event": "loading", "message": f"Loading dataset {dataset_name}..."}
            _jobs[job_id]["progress"] = loading_event
            progress_queue.put(loading_event)

            subset_cfg = config.get("longmemeval_subset")
            users_to_test = subset_cfg.get("users_to_test") if subset_cfg else None
            tenant_ids = subset_cfg.get("tenant_ids") if subset_cfg else None
            docs_by_tenant = load_longmemeval_docs_by_tenant(
                dataset_name, users_to_test=users_to_test, tenant_ids=tenant_ids,
            )

            total_sessions = sum(len(s) for s in docs_by_tenant.values())
            loaded_event = {
                "event": "loading",
                "message": f"Loaded {total_sessions} sessions across {len(docs_by_tenant)} tenants",
            }
            _jobs[job_id]["progress"] = loaded_event
            progress_queue.put(loaded_event)

            result = engram_ingest_all_tenants(
                docs_by_tenant, poll=True, on_progress=on_progress,
            )

            manifest = save_engram_manifest(result, dataset_name)

            # Build memory content→session index in background
            from query_agent_benchmarking.internal.adapters.results.engram_memory_index import build_and_save_memory_index
            try:
                progress_queue.put({"event": "loading", "message": "Building memory index..."})
                build_and_save_memory_index(result, dataset_name, manifest["timestamp"])
            except Exception as idx_err:
                # Non-fatal — index is a convenience, don't fail the whole job
                print(f"Warning: memory index build failed: {idx_err}")

            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["manifest"] = manifest
            progress_queue.put({"event": "complete", "manifest": manifest})
        except Exception as e:
            traceback.print_exc()
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            progress_queue.put({"event": "error", "message": str(e)})
        finally:
            progress_queue.put(None)  # sentinel

    def event_generator():
        # First event: tell the client its job_id so it can reconnect
        yield json.dumps({"event": "started", "job_id": job_id}) + "\n"

        thread = threading.Thread(target=run_ingestion, daemon=True)
        thread.start()

        while True:
            event = progress_queue.get()
            if event is None:
                break
            if "event" not in event:
                event = {"event": "progress", **event}
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )


@app.get("/populate-db-job/{job_id}")
def get_populate_job(job_id: str) -> dict[str, Any]:
    """Check the status of a running or completed populate job.

    Returns the latest progress event, and the manifest once complete.
    Used by the frontend to reconnect after navigating away.
    """
    job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "progress": job["progress"],
        "manifest": job["manifest"],
        "error": job["error"],
    }


@app.get("/engram-run-detail")
def engram_run_detail(
    run_id: str = Query(...),
    dataset_name: str = Query(...),
    tenant_id: str = Query(...),
    session_id: str = Query(...),
    user_id_prefix: str = Query("longmemeval-"),
    group: str = Query("default"),
) -> dict[str, Any]:
    """Fetch the session input text and memory operations for a single Engram run.

    Returns the original conversation text and the memories that were
    created, updated, or deleted by this run.
    """
    from engram import EngramClient
    from engram._http import APIError
    from query_agent_benchmarking.internal.adapters.dataset import load_longmemeval_docs_by_tenant

    try:
        # --- Load session text from dataset ---
        docs_by_tenant = load_longmemeval_docs_by_tenant(dataset_name)
        session_text = None
        tenant_docs = docs_by_tenant.get(tenant_id, [])
        for doc in tenant_docs:
            if doc.get("session_id") == session_id:
                session_text = doc.get("session_text", "")
                break

        # --- Fetch memory operations from Engram ---
        client = EngramClient(
            api_key=os.environ["ENGRAM_API_KEY"],
            base_url=os.environ.get("ENGRAM_BASE_URL", "https://dev-engram.labs.weaviate.io"),
        )
        user_id = f"{user_id_prefix}{tenant_id}"

        run_status = client.runs.get(run_id)
        ops = run_status.committed_operations

        def fetch_memories(op_list: list, op_type: str) -> list[dict]:
            results = []
            for op in (op_list or []):
                entry: dict[str, Any] = {
                    "memory_id": op.memory_id,
                    "operation": op_type,
                }
                try:
                    memory = client.memories.get(op.memory_id, user_id=user_id, group=group)
                    entry["content"] = memory.content
                    entry["created_at"] = str(getattr(memory, "created_at", ""))
                except APIError as e:
                    if e.status_code == 404:
                        entry["content"] = None
                        entry["note"] = "superseded by later run"
                    else:
                        raise
                results.append(entry)
            return results

        memories = (
            fetch_memories(getattr(ops, "created", []), "created")
            + fetch_memories(getattr(ops, "updated", []), "updated")
            + fetch_memories(getattr(ops, "deleted", []), "deleted")
        )

        return {
            "status": "ok",
            "session_text": session_text,
            "run_status": run_status.status,
            "memories": memories,
        }
    except Exception:
        traceback.print_exc()
        return {"status": "error", "error": "Failed to fetch run detail."}


@app.get("/memory-source-lookup")
def memory_source_lookup(
    content: str = Query(..., description="The memory content string to look up"),
) -> dict[str, Any]:
    """Look up which session produced a given memory by its content hash.

    Searches all saved memory index files for a matching content hash.
    Returns the source session info (tenant_id, session_id, operation history).
    """
    import hashlib
    from pathlib import Path

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    results_dir = (
        Path(__file__).resolve().parent.parent.parent.parent / "console" / "results"
    )
    if not results_dir.exists():
        return {"status": "not_found", "content_hash": content_hash}

    # Search all memory index files (newest first)
    index_files = sorted(
        [f for f in results_dir.iterdir() if f.name.startswith("engram-memory-index-") and f.name.endswith(".json")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    for index_file in index_files:
        try:
            with open(index_file) as f:
                index_doc = json.load(f)
            entry = index_doc.get("index", {}).get(content_hash)
            if entry:
                # Find the manifest that contains this run_id
                manifest_id = None
                run_id = entry["operations"][0]["run_id"] if entry.get("operations") else None
                if run_id:
                    manifest_files = sorted(
                        [f for f in results_dir.iterdir() if f.name.startswith("engram-ingest-") and f.name.endswith(".json")],
                        key=lambda f: f.stat().st_mtime,
                        reverse=True,
                    )
                    for mf in manifest_files:
                        try:
                            with open(mf) as mfh:
                                manifest_doc = json.load(mfh)
                            if any(r.get("run_id") == run_id for r in manifest_doc.get("runs", [])):
                                manifest_id = mf.name.replace(".json", "")
                                break
                        except Exception:
                            continue

                return {
                    "status": "found",
                    "content_hash": content_hash,
                    "memory_id": entry["memory_id"],
                    "content": entry["content"],
                    "operations": entry["operations"],
                    "index_file": index_file.name,
                    "dataset": index_doc.get("dataset"),
                    "manifest_id": manifest_id,
                }
        except Exception:
            continue

    return {"status": "not_found", "content_hash": content_hash}

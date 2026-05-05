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

import traceback
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import query_agent_benchmarking

app = FastAPI(
    title="Query Agent Benchmarking",
    description="Run Weaviate Query Agent benchmarks via HTTP",
)


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
        start = kwargs.pop("longmemeval_subset_start", None)
        end = kwargs.pop("longmemeval_subset_end", None)
        if start is not None and end is not None:
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


@app.post("/populate-db")
def populate_db_endpoint(req: PopulateDatabaseRequest) -> dict[str, Any]:
    """Populate the database from config, with optional overrides."""
    from pathlib import Path
    from query_agent_benchmarking.internal.config.loader import load_config

    try:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "internal" / "config" / "database_loader_config.yml"
        )
        config = load_config(str(config_path))

        # Apply overrides from request
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
        if req.longmemeval_subset_start is not None and req.longmemeval_subset_end is not None:
            config["longmemeval_subset"] = {
                "users_to_test": [req.longmemeval_subset_start, req.longmemeval_subset_end]
            }

        # Run population using the merged config
        from query_agent_benchmarking.internal.adapters.database.database_config import validate_database_dataset
        from query_agent_benchmarking.internal.core.services.populate_db import (
            _run_engram_loader,
            _run_weaviate_loader,
        )

        dataset_name = config["dataset_name"]
        validate_database_dataset(dataset_name)
        database_target = config.get("database_target", "weaviate")

        if database_target == "engram":
            _run_engram_loader(config, dataset_name)
        else:
            _run_weaviate_loader(config, dataset_name, recreate=req.recreate, tag=req.tag)

        return {"status": "ok"}
    except Exception:
        traceback.print_exc()
        return {"status": "error", "error": "An internal error occurred."}

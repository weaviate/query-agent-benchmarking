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
    num_trials: Optional[int] = None
    use_subset: Optional[bool] = None
    num_samples: Optional[int] = None
    use_async: Optional[bool] = None
    batch_size: Optional[int] = None
    max_concurrent: Optional[int] = None
    embedding_model: Optional[str] = None


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
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.post("/run-ask-benchmark")
def run_ask(req: RunAskBenchmarkRequest) -> dict[str, Any]:
    """Run an ask benchmark with the given parameters."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = query_agent_benchmarking.run_ask_eval(**kwargs)
        return {"status": "ok", "result": result}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.post("/compare-embeddings")
def run_compare(req: CompareEmbeddingsRequest) -> dict[str, Any]:
    """Compare embedding models."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = query_agent_benchmarking.compare_embeddings(**kwargs)
        return {"status": "ok", "result": result}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.post("/populate-db")
def populate_db(req: PopulateDatabaseRequest) -> dict[str, Any]:
    """Populate the database from config."""
    try:
        query_agent_benchmarking.database_loader(
            recreate=req.recreate,
            tag=req.tag,
        )
        return {"status": "ok"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

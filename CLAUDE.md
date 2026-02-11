# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python library for benchmarking Weaviate's Query Agent. It supports two evaluation modes:
- **Search mode**: Ranked retrieval evaluation using IR metrics (Recall@K, nDCG@K, Coverage, alpha-nDCG)
- **Ask mode**: Question answering evaluation using LLM-as-judge (DSPy-based ensemble voting for semantic alignment)

## Commands

```bash
# Install dependencies
uv sync

# Populate Weaviate with benchmark data
uv run python scripts/populate-db.py

# Run search benchmark
uv run python scripts/run-search-benchmark.py

# Run ask benchmark
uv run python scripts/run-ask-benchmark.py

# Run embedding model comparison
uv run python scripts/run-compare-embeddings.py
```

No test suite exists in this repository.

## Environment Variables

Requires `WEAVIATE_URL`, `WEAVIATE_API_KEY`, and `OPENAI_API_KEY`. Third-party embedding providers may need `COHERE_API_KEY` or `VOYAGEAI_API_KEY`.

## Architecture

### Entry Points

The package exposes three main functions via `__init__.py`:
- `run_search_eval()` / `run_search_evals()` - in `search_benchmark_run.py`
- `run_ask_eval()` - in `ask_benchmark_run.py`
- `compare_embeddings()` - in `compare_embeddings.py`

All accept either programmatic kwargs or load from YAML config files (`benchmark-config.yml` for benchmarks, `database/database_loader_config.yml` for DB population). Kwargs override file config via `merge_configs()`.

### Agent Layer (`agent/`)

`BaseAgentBuilder` (ABC) handles Weaviate connection and dataset-to-collection name mapping. Two concrete builders:
- `SearchAgentBuilder`: Wraps `QueryAgent` (search-only mode), Weaviate hybrid search, or external HTTP service
- `AskAgentBuilder`: Wraps `QueryAgent` (ask mode) or external HTTP service

External service mode sends POST requests to a configurable host, enabling BYOS (bring-your-own-system) evaluation. Expected formats:
- Search: `{"query": "..."}` -> `{"results": ["id1", "id2", ...]}`
- Ask: `{"question": "...", "oracle_context_id": "..."}` -> `{"answer": "..."}`

### Data Flow

1. **Dataset loading** (`dataset.py`): Loads queries/docs from HuggingFace Hub or Weaviate collections into `InMemoryQuery`/`InMemoryAskQuery` Pydantic models
2. **DB population** (`database/`): Registry-based system — `database_registry.py` defines `DatasetSpec` per dataset using a builder pattern (`DatasetSpecBuilder`), `database_loader.py` creates Weaviate collections with batch insert
3. **Query execution** (`query_agent_benchmark.py`): Runs queries (sync or async with batching/semaphore concurrency), produces `QueryResult`/`AskResult`
4. **Metrics** (`metrics/`): IR metrics in `ir_metrics.py`; LLM judge in `lmjudge_alignment.py` using DSPy `Predict` with ensemble voting
5. **Serialization** (`result_serialization.py`): Saves per-trial results, per-trial metrics, and aggregated cross-trial results as JSON

### Dataset-Metric Mapping

Different datasets use different metrics (configured in `query_agent_benchmark.py:analyze_search_results`):
- BEIR/BRIGHT: Recall@1/5/20, nDCG@10
- FreshStack: Recall@50, Coverage@5/10/20, alpha-nDCG@10
- LoTTe: Recall@1/5/20, Success@5

### Collection Naming Convention

Built-in datasets map to Weaviate collections as `{DatasetPrefix}{PascalizedSubset}_{Tag}` (e.g., `FreshstackLangchain_Default`, `BeirScifact_Default`). The tag defaults to "Default" and supports aliasing.

## Key Pydantic Models (`models.py`)

- `InMemoryQuery` / `InMemorySearchQuery`: Search queries with `dataset_ids` ground truth and optional FreshStack nugget data
- `InMemoryAskQuery`: Ask queries with `ground_truth_answer` and optional `oracle_context_id`
- `DocsCollection` / `QueriesCollection` / `AskQueriesCollection`: Custom collection configs for non-builtin datasets

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

# Run tests
uv run pytest tests/ -v

# Populate Weaviate with benchmark data
uv run python scripts/populate-db.py

# Run search benchmark
uv run python scripts/run-search-benchmark.py

# Run ask benchmark
uv run python scripts/run-ask-benchmark.py

# Run embedding model comparison
uv run python scripts/run-compare-embeddings.py
```

## Environment Variables

Requires `WEAVIATE_URL`, `WEAVIATE_API_KEY`, and `OPENAI_API_KEY`. Third-party embedding providers may need `COHERE_API_KEY` or `VOYAGEAI_API_KEY`.

## Architecture

The codebase follows **hexagonal architecture** (ports & adapters) to separate domain logic from infrastructure concerns.

### Directory Structure

```
query_agent_benchmarking/
├── domain/                    # CORE - no external dependencies
│   ├── models.py              # Pydantic models (canonical location)
│   ├── metrics_config.py      # MetricSpec, MetricsProfile, dataset-metrics registry (data only)
│   ├── query_execution.py     # run_search_queries[_async], run_ask_queries[_async]
│   ├── analysis.py            # aggregate_metrics (pure math)
│   └── benchmark_orchestrator.py  # DI-based orchestrators for search/ask flows
│
├── ports/                     # INTERFACES - Python Protocols
│   ├── search_agent.py        # SearchAgent protocol
│   ├── ask_agent.py           # AskAgent protocol + AskResponse
│   ├── dataset_repository.py  # SearchDatasetRepository, AskDatasetRepository
│   ├── metrics_calculator.py  # SearchMetricsCalculator, AskMetricsCalculator
│   ├── result_repository.py   # ResultRepository protocol
│   ├── database_manager.py    # DatabaseManager protocol
│   └── llm_judge.py           # LLMJudge protocol
│
├── adapters/                  # IMPLEMENTATIONS
│   ├── agents/                # SearchAgent/AskAgent adapters
│   │   ├── weaviate_query_agent.py  # WeaviateQueryAgentSearch, WeaviateQueryAgentAsk
│   │   ├── weaviate_search.py       # WeaviateHybridSearch, WeaviateVectorSearch, WeaviateBM25Search
│   │   ├── external_service.py      # ExternalSearchService, ExternalAskService
│   │   └── collection_resolver.py   # Dataset-to-collection name mapping
│   ├── clients/               # Cross-cutting infrastructure
│   │   ├── weaviate_client.py       # Unified client factory
│   │   └── provider_headers.py      # Single canonical provider normalization
│   ├── dataset/               # DatasetRepository adapters
│   │   ├── huggingface_loader.py    # HF Hub datasets
│   │   ├── ir_datasets_loader.py    # BEIR, LOTTE via ir_datasets
│   │   ├── weaviate_loader.py       # Queries from Weaviate collections
│   │   ├── local_file_loader.py     # OfficeQA PDFs/CSV
│   │   └── registry.py             # Dataset name dispatcher
│   ├── metrics/               # MetricsCalculator adapters
│   │   ├── ir_metrics_calculator.py     # IRMetricsCalculator (SearchMetricsCalculator)
│   │   └── ask_metrics_calculator.py    # LMJudge, ExactMatch, OfficeQA calculators
│   └── results/
│       └── json_file_repository.py  # JSON file I/O (ResultRepository)
│
├── agent/                     # Legacy agent builders (backward-compatible)
├── metrics/                   # Raw metric functions (ir_metrics, lmjudge, exact_match)
├── database/                  # DB population (registry, loader, specs)
├── search_benchmark_run.py    # Search entry point (config -> adapters -> domain)
├── ask_benchmark_run.py       # Ask entry point (config -> adapters -> domain)
├── query_agent_benchmark.py   # Backward-compatible facade
├── dataset.py                 # Backward-compatible facade -> adapters/dataset/
├── models.py                  # Backward-compatible re-exports -> domain/models
├── result_serialization.py    # Original file I/O (wrapped by json_file_repository)
└── compare_embeddings.py      # Embedding model comparison
```

### Entry Points

The package exposes three main functions via `__init__.py`:
- `run_search_eval()` / `run_search_evals()` - in `search_benchmark_run.py`
- `run_ask_eval()` - in `ask_benchmark_run.py`
- `compare_embeddings()` - in `compare_embeddings.py`

All accept either programmatic kwargs or load from YAML config files (`benchmark-config.yml` for benchmarks, `database/database_loader_config.yml` for DB population). Kwargs override file config via `merge_configs()`.

### Domain Layer

The domain layer has **no external dependencies** (no Weaviate, HuggingFace, DSPy imports):
- `models.py`: All Pydantic models (ObjectID, InMemoryQuery, QueryResult, AskResult, etc.)
- `metrics_config.py`: `MetricSpec` (data-only metric spec) + `DATASET_METRICS_REGISTRY` mapping datasets to metrics
- `query_execution.py`: Sync/async query runners accepting port-typed agents
- `analysis.py`: `aggregate_metrics()` for cross-trial statistical aggregation
- `benchmark_orchestrator.py`: `SearchBenchmarkOrchestrator` / `AskBenchmarkOrchestrator` with DI

### Agent Layer (`agent/`)

`BaseAgentBuilder` (ABC) handles Weaviate connection and dataset-to-collection name mapping. Two concrete builders:
- `SearchAgentBuilder`: Wraps `QueryAgent` (search-only mode), Weaviate hybrid search, or external HTTP service
- `AskAgentBuilder`: Wraps `QueryAgent` (ask mode) or external HTTP service

Clean protocol-implementing alternatives are in `adapters/agents/`:
- `WeaviateQueryAgentSearch` / `WeaviateQueryAgentAsk`: Clean wrappers for QueryAgent
- `WeaviateHybridSearch`, `WeaviateVectorSearch`, `WeaviateBM25Search`: Direct search
- `ExternalSearchService` / `ExternalAskService`: HTTP-based BYOS adapters

External service mode sends POST requests to a configurable host, enabling BYOS (bring-your-own-system) evaluation. Expected formats:
- Search: `{"query": "..."}` -> `{"results": ["id1", "id2", ...]}`
- Ask: `{"question": "...", "oracle_context_id": "..."}` -> `{"answer": "..."}`

### Data Flow

1. **Dataset loading** (`adapters/dataset/`): Loads queries/docs from HuggingFace Hub, ir_datasets, or Weaviate collections into `InMemoryQuery`/`InMemoryAskQuery` Pydantic models. `dataset.py` is a backward-compatible facade.
2. **DB population** (`database/`): Registry-based system — `database_registry.py` defines `DatasetSpec` per dataset using a builder pattern (`DatasetSpecBuilder`), `database_loader.py` creates Weaviate collections with batch insert
3. **Query execution** (`domain/query_execution.py`): Runs queries (sync or async with batching/semaphore concurrency), produces `QueryResult`/`AskResult`
4. **Metrics** (`adapters/metrics/`): `IRMetricsCalculator` dispatches `MetricSpec` names to functions in `metrics/ir_metrics.py`. Ask metrics via `LMJudgeAskCalculator`, `ExactMatchAskCalculator`, or `OfficeQAAskCalculator`.
5. **Serialization** (`adapters/results/json_file_repository.py`): `JsonFileResultRepository` wraps `result_serialization.py` for per-trial and aggregated JSON output

### Dataset-Metric Mapping

Different datasets use different metrics (configured in `domain/metrics_config.py`):
- BEIR/BRIGHT: Recall@1/5/20, nDCG@10
- FreshStack: Recall@50, Coverage@5/10/20, alpha-nDCG@10
- LoTTe: Recall@1/5/20, Success@5
- LongMemEval: Recall@1/5/10, nDCG@10

### Collection Naming Convention

Built-in datasets map to Weaviate collections as `{DatasetPrefix}{PascalizedSubset}_{Tag}` (e.g., `FreshstackLangchain_Default`, `BeirScifact_Default`). The tag defaults to "Default" and supports aliasing.

## Key Pydantic Models (`domain/models.py`)

- `InMemoryQuery` / `InMemorySearchQuery`: Search queries with `dataset_ids` ground truth and optional FreshStack nugget data
- `InMemoryAskQuery`: Ask queries with `ground_truth_answer` and optional `oracle_context_id`
- `DocsCollection` / `QueriesCollection` / `AskQueriesCollection`: Custom collection configs for non-builtin datasets

## Tests

```bash
uv run pytest tests/ -v                    # All tests
uv run pytest tests/domain/ -v             # Domain logic tests
uv run pytest tests/adapters/ -v           # Adapter tests
uv run pytest tests/integration/ -v        # End-to-end integration tests
```

Test structure:
- `tests/domain/`: MetricSpec, aggregate_metrics, query execution, orchestrators
- `tests/adapters/`: IR metrics, exact match, JSON repository, agent adapters, parsing utilities
- `tests/integration/`: Full search/ask pipeline E2E tests with mock agents

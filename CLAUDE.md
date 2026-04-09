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

The codebase follows **hexagonal architecture** (ports & adapters) with a `cmd/` + `internal/` layout separating entry points from library code.

### Directory Structure

```
query_agent_benchmarking/
├── cmd/                           # Entry point orchestration
│   ├── run_search.py              # Search benchmark runner
│   ├── run_ask.py                 # Ask benchmark runner
│   └── run_compare_embeddings.py  # Embedding model comparison
│
├── internal/                      # Library internals
│   ├── core/                      # CORE - no external dependencies
│   │   ├── models.py              # Pydantic models (canonical location)
│   │   ├── metrics_config.py      # MetricSpec, MetricsProfile, dataset-metrics registry
│   │   ├── query_execution.py     # run_search_queries[_async], run_ask_queries[_async]
│   │   ├── analysis.py            # aggregate_metrics (pure math)
│   │   ├── benchmark_orchestrator.py  # DI-based orchestrators
│   │   └── ports/                 # Python Protocol interfaces
│   │       ├── search_agent.py    # SearchAgent protocol
│   │       ├── ask_agent.py       # AskAgent protocol + AskResponse
│   │       ├── dataset_repository.py
│   │       ├── metrics_calculator.py
│   │       ├── result_repository.py
│   │       ├── database_manager.py
│   │       └── llm_judge.py
│   │
│   ├── adapters/                  # Port implementations
│   │   ├── agents/                # SearchAgent/AskAgent adapters
│   │   │   ├── weaviate_query_agent.py
│   │   │   ├── weaviate_search.py
│   │   │   ├── external_service.py
│   │   │   └── collection_resolver.py
│   │   ├── clients/               # Cross-cutting infrastructure
│   │   │   ├── weaviate_client.py
│   │   │   └── provider_headers.py
│   │   ├── dataset/               # DatasetRepository adapters
│   │   │   ├── huggingface_loader.py
│   │   │   ├── ir_datasets_loader.py
│   │   │   ├── weaviate_loader.py
│   │   │   ├── local_file_loader.py
│   │   │   └── registry.py
│   │   ├── metrics/               # MetricsCalculator adapters
│   │   │   ├── ir_metrics_calculator.py
│   │   │   └── ask_metrics_calculator.py
│   │   └── results/
│   │       └── json_file_repository.py
│   │
│   ├── config/                    # Configuration
│   │   ├── config.py              # Dataset lists, named vector targets
│   │   └── qa_system_prompt_registry.py
│   │
│   ├── mocks/                     # Mock implementations for testing
│   │   ├── agents.py              # MockSearchAgent, MockAskAgent
│   │   └── repositories.py        # MockResultRepository
│   │
│   └── testutil/                  # Test utilities and factories
│       └── factories.py           # make_search_queries(), make_ask_results(), etc.
│
├── agent/                         # Legacy agent builders (backward-compatible)
├── metrics/                       # Raw metric functions
├── database/                      # DB population (registry, loader, specs)
├── __init__.py                    # Public API (all exports preserved)
├── models.py                      # Re-exports -> internal/core/models
├── config.py                      # Re-exports -> internal/config/config
├── dataset.py                     # Facade -> internal/adapters/dataset/
├── query_agent_benchmark.py       # Backward-compatible facade
├── domain/                        # Re-exports -> internal/core/
├── ports/                         # Re-exports -> internal/core/ports/
└── adapters/                      # Re-exports -> internal/adapters/
```

### Entry Points

The package exposes three main functions via `__init__.py`:
- `run_search_eval()` / `run_search_evals()` - in `cmd/run_search.py`
- `run_ask_eval()` - in `cmd/run_ask.py`
- `compare_embeddings()` - in `cmd/run_compare_embeddings.py`

All accept either programmatic kwargs or load from YAML config files (`benchmark-config.yml` for benchmarks, `database/database_loader_config.yml` for DB population). Kwargs override file config via `merge_configs()`.

### Core Layer (`internal/core/`)

The core layer has **no external dependencies** (no Weaviate, HuggingFace, DSPy imports):
- `models.py`: All Pydantic models (ObjectID, InMemoryQuery, QueryResult, AskResult, etc.)
- `metrics_config.py`: `MetricSpec` (data-only metric spec) + `DATASET_METRICS_REGISTRY` mapping datasets to metrics
- `query_execution.py`: Sync/async query runners accepting port-typed agents
- `analysis.py`: `aggregate_metrics()` for cross-trial statistical aggregation
- `benchmark_orchestrator.py`: `SearchBenchmarkOrchestrator` / `AskBenchmarkOrchestrator` with DI
- `ports/`: Python Protocol interfaces for all boundaries

### Agent Layer (`agent/`)

`BaseAgentBuilder` (ABC) handles Weaviate connection and dataset-to-collection name mapping. Two concrete builders:
- `SearchAgentBuilder`: Wraps `QueryAgent` (search-only mode), Weaviate hybrid search, or external HTTP service
- `AskAgentBuilder`: Wraps `QueryAgent` (ask mode) or external HTTP service

Clean protocol-implementing alternatives are in `internal/adapters/agents/`:
- `WeaviateQueryAgentSearch` / `WeaviateQueryAgentAsk`: Clean wrappers for QueryAgent
- `WeaviateHybridSearch`, `WeaviateVectorSearch`, `WeaviateBM25Search`: Direct search
- `ExternalSearchService` / `ExternalAskService`: HTTP-based BYOS adapters

### Data Flow

1. **Dataset loading** (`internal/adapters/dataset/`): Loads queries/docs from HuggingFace Hub, ir_datasets, or Weaviate collections into `InMemoryQuery`/`InMemoryAskQuery` Pydantic models
2. **DB population** (`database/`): Registry-based system — `database_registry.py` defines `DatasetSpec` per dataset
3. **Query execution** (`internal/core/query_execution.py`): Runs queries (sync or async with batching/semaphore concurrency)
4. **Metrics** (`internal/adapters/metrics/`): `IRMetricsCalculator` dispatches `MetricSpec` names to metric functions. Ask metrics via `LMJudgeAskCalculator`, `ExactMatchAskCalculator`, or `OfficeQAAskCalculator`
5. **Serialization** (`internal/adapters/results/json_file_repository.py`): `JsonFileResultRepository` for per-trial and aggregated JSON output

### Backward Compatibility

Old import paths are preserved via re-export facades at `domain/`, `ports/`, `adapters/`, `config.py`, `search_benchmark_run.py`, `ask_benchmark_run.py`, and `compare_embeddings.py`. All existing `from query_agent_benchmarking.domain.models import ...` style imports continue to work.

### Dataset-Metric Mapping

Different datasets use different metrics (configured in `internal/core/metrics_config.py`):
- BEIR/BRIGHT: Recall@1/5/20, nDCG@10
- FreshStack: Recall@50, Coverage@5/10/20, alpha-nDCG@10
- LoTTe: Recall@1/5/20, Success@5
- LongMemEval: Recall@1/5/10, nDCG@10

### Collection Naming Convention

Built-in datasets map to Weaviate collections as `{DatasetPrefix}{PascalizedSubset}_{Tag}` (e.g., `FreshstackLangchain_Default`, `BeirScifact_Default`). The tag defaults to "Default" and supports aliasing.

## Key Pydantic Models (`internal/core/models.py`)

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
- Shared fixtures in `tests/conftest.py` use `internal/mocks/` and `internal/testutil/`

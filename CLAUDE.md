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

The codebase follows **hexagonal architecture** (ports & adapters) with a `cmd/` + `internal/` layout.

### Directory Structure

```
query_agent_benchmarking/
├── __init__.py                        # Public API exports
├── benchmark-config.yml               # Default benchmark configuration
│
├── cmd/                               # Entry point orchestration
│   ├── run_search.py                  # Search benchmark runner
│   ├── run_ask.py                     # Ask benchmark runner
│   └── run_compare_embeddings.py      # Embedding model comparison
│
├── internal/                          # Library internals
│   ├── core/                          # Domain logic (no external deps)
│   │   ├── models.py                  # Pydantic models
│   │   ├── metrics_config.py          # MetricSpec, dataset-metrics registry
│   │   ├── query_execution.py         # run_search/ask_queries[_async]
│   │   ├── analysis.py               # aggregate_metrics
│   │   ├── benchmark_orchestrator.py  # DI-based orchestrators
│   │   └── ports/                     # Python Protocol interfaces
│   │       ├── search_agent.py
│   │       ├── ask_agent.py
│   │       ├── dataset_repository.py
│   │       ├── metrics_calculator.py
│   │       ├── result_repository.py
│   │       ├── database_manager.py
│   │       └── llm_judge.py
│   │
│   ├── adapters/                      # Port implementations
│   │   ├── agents/                    # Protocol-based agent adapters
│   │   │   ├── weaviate_query_agent.py
│   │   │   ├── weaviate_search.py
│   │   │   ├── external_service.py
│   │   │   └── collection_resolver.py
│   │   ├── clients/                   # Infrastructure clients
│   │   │   ├── weaviate_client.py
│   │   │   └── provider_headers.py
│   │   ├── database/                  # DB population & schema
│   │   │   ├── database_registry.py
│   │   │   ├── database_loader.py
│   │   │   ├── property_builder.py
│   │   │   ├── spec.py
│   │   │   └── ...
│   │   ├── dataset/                   # Dataset loaders
│   │   │   ├── huggingface_loader.py
│   │   │   ├── ir_datasets_loader.py
│   │   │   ├── weaviate_loader.py
│   │   │   ├── local_file_loader.py
│   │   │   └── registry.py
│   │   ├── metrics/                   # Metric implementations
│   │   │   ├── ir_metrics.py
│   │   │   ├── ir_metrics_calculator.py
│   │   │   ├── ask_metrics_calculator.py
│   │   │   ├── lmjudge_alignment.py
│   │   │   ├── exact_match.py
│   │   │   └── officeqa_metric.py
│   │   └── results/
│   │       ├── json_file_repository.py
│   │       └── serialization.py
│   │
│   ├── agents/                        # Legacy agent builders
│   │   ├── base.py                    # BaseAgentBuilder ABC
│   │   ├── search_agent.py            # SearchAgentBuilder
│   │   ├── ask_agent.py               # AskAgentBuilder
│   │   └── engram_dspy_agent.py       # EngramDSPyAgent
│   │
│   ├── config/                        # Configuration
│   │   ├── config.py                  # Dataset lists, named vector targets
│   │   └── qa_system_prompt_registry.py
│   │
│   ├── mocks/                         # Mock implementations for testing
│   │   ├── agents.py
│   │   └── repositories.py
│   │
│   ├── testutil/                      # Test factories
│   │   └── factories.py
│   │
│   ├── dataset.py                     # Dataset loading facade
│   └── utils.py                       # Shared utilities
│
└── experimental/                      # Experimental tools
```

### Entry Points

The package exposes three main functions via `__init__.py`:
- `run_search_eval()` / `run_search_evals()` - in `cmd/run_search.py`
- `run_ask_eval()` - in `cmd/run_ask.py`
- `compare_embeddings()` - in `cmd/run_compare_embeddings.py`

All accept either programmatic kwargs or load from YAML config files. Kwargs override file config via `merge_configs()`.

### Core Layer (`internal/core/`)

No external dependencies (no Weaviate, HuggingFace, DSPy imports):
- `models.py`: All Pydantic models (ObjectID, InMemoryQuery, QueryResult, AskResult, etc.)
- `metrics_config.py`: `MetricSpec` + `DATASET_METRICS_REGISTRY` mapping datasets to metrics
- `query_execution.py`: Sync/async query runners accepting port-typed agents
- `analysis.py`: `aggregate_metrics()` for cross-trial statistical aggregation
- `benchmark_orchestrator.py`: `SearchBenchmarkOrchestrator` / `AskBenchmarkOrchestrator` with DI
- `ports/`: Python Protocol interfaces for all boundaries

### Data Flow

1. **Dataset loading** (`internal/adapters/dataset/`): Loads from HuggingFace Hub, ir_datasets, or Weaviate
2. **DB population** (`internal/adapters/database/`): Registry-based collection creation
3. **Query execution** (`internal/core/query_execution.py`): Sync/async with batching/semaphore
4. **Metrics** (`internal/adapters/metrics/`): IR metrics, LLM judge, exact match, OfficeQA
5. **Serialization** (`internal/adapters/results/`): JSON file I/O

### Dataset-Metric Mapping

Configured in `internal/core/metrics_config.py`:
- BEIR/BRIGHT: Recall@1/5/20, nDCG@10
- FreshStack: Recall@50, Coverage@5/10/20, alpha-nDCG@10
- LoTTe: Recall@1/5/20, Success@5
- LongMemEval: Recall@1/5/10, nDCG@10

### Collection Naming Convention

Built-in datasets map to Weaviate collections as `{DatasetPrefix}{PascalizedSubset}_{Tag}` (e.g., `FreshstackLangchain_Default`, `BeirScifact_Default`).

## Tests

```bash
uv run pytest tests/ -v                    # All tests
uv run pytest tests/domain/ -v             # Domain logic tests
uv run pytest tests/adapters/ -v           # Adapter tests
uv run pytest tests/integration/ -v        # End-to-end integration tests
```

- `tests/domain/`: MetricSpec, aggregate_metrics, query execution, orchestrators
- `tests/adapters/`: IR metrics, exact match, JSON repository, agent adapters
- `tests/integration/`: Full search/ask pipeline E2E tests with mock agents
- Shared fixtures in `tests/conftest.py` use `internal/mocks/` and `internal/testutil/`

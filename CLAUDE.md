# CLAUDE.md (AI-generated overview)

## Project Overview

A Python library for benchmarking Weaviate's Query Agent. It supports two evaluation modes:
- **Search mode**: Ranked retrieval evaluation using IR metrics (Recall@K, nDCG@K, Coverage, alpha-nDCG)
- **Ask mode**: Question answering evaluation using LLM-as-judge (DSPy-based ensemble voting for semantic alignment)

## Commands to Run

```bash
uv sync                                        # Install dependencies
uv run python3 scripts/populate-db.py           # Populate Weaviate with benchmark data
uv run python3 scripts/run-search-benchmark.py  # Run search benchmark
uv run python3 scripts/run-ask-benchmark.py     # Run ask benchmark
uv run python3 scripts/run-compare-embeddings.py # Compare embedding models

# Start the toy benchmark server
uv run uvicorn query_agent_benchmarking.cmd.server:app --reload

# Run tests (unit/integration only — fast, no network)
uv run pytest -v

# Run functional tests (downloads real data)
uv run pytest tests/functional/ -v

# Run Weaviate integration tests (requires WEAVIATE_URL + WEAVIATE_API_KEY)
uv run pytest tests/functional/test_database_weaviate.py -v
```

## Environment Variables

Configured via `.env` at the project root (loaded automatically via `python-dotenv`). See `.env.example` for the full list.

---

## Hexagonal Architecture

This package is organized using the **hexagonal architecture** pattern, also known as "ports and adapters." The central idea is that the domain logic — the rules, models, and workflows that define what the system *does* — should never depend on infrastructure details like which database you're talking to, which HTTP client you're using, or how files get written to disk. Instead, the domain declares abstract interfaces (called **ports**) that describe what capabilities it *needs*, and concrete implementations (called **adapters**) plug into those ports from the outside. This inverts the traditional dependency direction: infrastructure depends on the domain, never the other way around. The payoff is that the core logic becomes testable in isolation (swap in a mock adapter), extensible without modification (add a new adapter for a new database), and readable on its own terms (the domain code says *what* without drowning in *how*).

### How It Works in This Package

The package is split into three layers inside `query_agent_benchmarking/internal/`:

```
internal/
├── core/           # The domain — pure logic, no infrastructure imports
│   ├── models.py           # Pydantic data contracts (InMemoryQuery, QueryResult, etc.)
│   ├── ports/              # Abstract interfaces the domain needs
│   ├── metrics_config.py   # Dataset-to-metrics mapping (data only, no function refs)
│   ├── query_execution.py  # Query runner logic (iteration, batching, concurrency)
│   ├── analysis.py         # Cross-trial metric aggregation (pure math)
│   ├── benchmark_orchestrator.py  # DI-wired orchestrators
│   └── services/           # Application-level entry points
│       ├── search_benchmark.py
│       ├── ask_benchmark.py
│       └── compare_embeddings.py
│
├── adapters/       # Concrete implementations that plug into ports
│   ├── agents/         # SearchAgent & AskAgent implementations
│   ├── clients/        # Weaviate client factory, provider header resolution
│   ├── database/       # Collection creation, batch insert, dataset specs
│   ├── dataset/        # Data loaders (HuggingFace, ir_datasets, Weaviate, local)
│   ├── metrics/        # IR metrics, LLM judge, exact match, OfficeQA
│   └── results/        # JSON file persistence
│
├── config/         # YAML config loading, dataset/metric registries, system prompts
├── agents/         # Agent builder factories (create the right adapter from config)
├── mocks/          # No-op implementations for testing
└── testutil/       # Query/result factory helpers for tests
```

### Ports

The domain declares seven port protocols in `core/ports/`, each a Python `Protocol` class:

| Port | What it abstracts | Adapters |
|---|---|---|
| `SearchAgent` | Executing a search query and returning ranked document IDs | Weaviate QueryAgent, hybrid/vector/BM25 search, external HTTP service |
| `AskAgent` | Executing a question and returning a natural-language answer | Weaviate QueryAgent (ask mode), external HTTP service, Engram+DSPy |
| `SearchDatasetRepository` | Loading search queries and corpus documents | HuggingFace Hub, ir_datasets (BEIR/LoTTe), Weaviate collections |
| `AskDatasetRepository` | Loading ask queries with ground-truth answers | HuggingFace Hub, Weaviate collections |
| `SearchMetricsCalculator` | Computing search metrics from results | IR metrics calculator (Recall, nDCG, Coverage, alpha-nDCG, Success) |
| `AskMetricsCalculator` | Computing ask metrics from results | LLM judge, exact match, OfficeQA fuzzy match |
| `ResultRepository` | Persisting trial results and aggregated metrics | JSON file repository |
| `LLMJudge` | Evaluating semantic alignment between answers | DSPy ensemble voting judge |
| `DatabaseManager` | Creating collections and batch-inserting documents | Weaviate collection manager |

### Data Flow

1. **Configuration** (`config/`): YAML files are loaded and merged with programmatic kwargs.
2. **Dataset loading** (`adapters/dataset/`): Queries and corpus are loaded into Pydantic models (`InMemoryQuery`, `InMemoryAskQuery`) via the dataset registry.
3. **Agent construction** (`agents/`): Builder factories read config and instantiate the right `SearchAgent` or `AskAgent` adapter.
4. **Query execution** (`core/query_execution.py`): Queries are run through the agent (sync or async with semaphore concurrency), producing `QueryResult` or `AskResult` objects.
5. **Metrics** (`adapters/metrics/`): A `MetricsCalculator` adapter computes scores. Which metrics to use is determined by `core/metrics_config.py` based on dataset name patterns.
6. **Persistence** (`adapters/results/`): A `ResultRepository` adapter saves per-trial results, per-trial metrics, and cross-trial aggregations.

### Testing Without Infrastructure

Because the domain depends only on protocols, the entire benchmark pipeline can be tested without Weaviate, HuggingFace, or any LLM:

```python
orchestrator = SearchBenchmarkOrchestrator(
    query_runner=mock_search_agent,       # returns fixed IDs
    metrics_calculator=IRMetricsCalculator(dataset_name="beir/scifact"),
    result_repository=MockResultRepository(),  # stores in memory
)
metrics = orchestrator.run_and_aggregate(queries, num_trials=3)
```

### Dataset-Metric Mapping

Different datasets use different metrics, configured as pure data in `core/metrics_config.py`:

| Dataset family | Metrics |
|---|---|
| BEIR, BRIGHT | Recall@1, Recall@5, Recall@20, nDCG@10 |
| FreshStack | Recall@50, Coverage@5/10/20, alpha-nDCG@10 |
| LoTTe | Recall@1, Recall@5, Recall@20, Success@5 |

### Collection Naming Convention

Built-in datasets map to Weaviate collections as `{DatasetPrefix}{PascalizedSubset}_{Tag}` (e.g., `FreshstackLangchain_Default`, `BeirScifact_Default`). The tag defaults to `"Default"` and supports aliasing for blue-green deployments.

## Top-Level Layout

```
query_agent_benchmarking/
├── __init__.py          # Public API surface
├── cmd/                 # Server (uvicorn entry point)
│   └── server.py
├── internal/            # All package internals (core + adapters + config)
└── experimental/        # Benchmark creation, hard negatives

scripts/                 # Runnable entry points (uv run python3 scripts/...)
tests/                   # pytest suite
├── functional/          # Real data & live Weaviate tests (opt-in)
└── ...                  # Unit/integration tests (fast, no network)
```

## Testing

The test suite is organized into three tiers:

### Tier 1: Unit & Integration Tests (default)

```bash
uv run pytest                  # ~103 tests, no network or credentials needed
```

Fast tests that use mocks and in-memory fixtures. These run by default — functional tests are excluded via `pyproject.toml` (`addopts = "--ignore=tests/functional"`).

### Tier 2: Functional Tests — Dataset Loading & Pipeline Validation

```bash
uv run pytest tests/functional/ -v                        # all functional tests
uv run pytest tests/functional/test_dataset_loading.py -v  # dataset loading only
uv run pytest tests/functional/test_database_loading.py -v # database pipeline only
```

These download real data from HuggingFace Hub and ir_datasets to verify the full loading pipeline:

- **`test_dataset_loading.py`** — Loads every supported search and ask dataset, verifies corpus docs have ID fields, queries are well-formed `InMemoryQuery`/`InMemoryAskQuery` instances, and both `corpus_only` and `queries_only` flags work correctly. Covers 11 search datasets and 3 ask datasets.

- **`test_database_loading.py`** — Validates the database loading pipeline *without* a live Weaviate instance. For each dataset: resolves the `DatasetSpec` from the registry, verifies `name_fn` produces a collection name, maps documents through `item_to_props`, checks that mapped keys are a subset of the declared schema properties, and confirms every doc includes a `dataset_id`. Catches schema mismatches early.

### Tier 3: Functional Tests — Live Weaviate Integration

```bash
uv run pytest tests/functional/test_database_weaviate.py -v
```

Requires `WEAVIATE_URL` and `WEAVIATE_API_KEY` environment variables. **Auto-skips** if credentials are not set.

Creates temporary collections (tagged `_FuncTest`), inserts 5 sample documents per dataset, verifies the count via aggregate query, then cleans up. Tests both standard and multi-tenant (LongMemEval) collection types. Covers 8 standard datasets and 1 multi-tenant dataset.

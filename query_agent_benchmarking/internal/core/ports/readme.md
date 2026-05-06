# Ports

Ports are abstract interfaces (Python `Protocol` classes) that define what capabilities the domain layer needs without specifying how they are implemented. Concrete implementations live in `adapters/` and plug into these protocols, keeping the domain free of infrastructure dependencies.

## Files and Protocols

### `search_agent.py` — `SearchAgent`

Interface for executing a search query and returning ranked document IDs. Supports both sync (`run`) and async (`run_async`) execution, with optional `tenant` parameter for multi-tenant datasets. Async lifecycle is managed via `initialize_async` and `close_async`.

**Adapters:** Weaviate QueryAgent, hybrid/vector/BM25 search, external HTTP service.

### `ask_agent.py` — `AskAgent`, `AskResponse`

Interface for executing a question and returning a natural-language answer wrapped in an `AskResponse` dataclass. Like `SearchAgent`, supports sync and async execution with optional `oracle_context_id` (for oracle context lookup) and `tenant_id` (for multi-tenant datasets).

**Adapters:** Weaviate QueryAgent (ask mode), external HTTP service, Engram+DSPy.

### `dataset_repository.py` — `SearchDatasetRepository`, `AskDatasetRepository`

Interfaces for loading benchmark datasets. `SearchDatasetRepository` provides `load_queries` (returns `InMemoryQuery` with ground-truth document IDs) and `load_corpus` (returns raw document dicts). `AskDatasetRepository` provides `load_queries` (returns `InMemoryAskQuery` with ground-truth answers).

**Adapters:** HuggingFace Hub, ir_datasets (BEIR/LoTTe), Weaviate collections, local JSON files.

### `metrics_calculator.py` — `SearchMetricsCalculator`, `AskMetricsCalculator`

Interfaces for scoring benchmark results. `SearchMetricsCalculator.compute` takes query results and ground truths, returning a metrics dictionary (e.g., Recall@K, nDCG@10). `AskMetricsCalculator.compute` takes ask results and returns a metrics dictionary (e.g., accuracy, per-type breakdown).

**Adapters:** IR metrics calculator, LLM judge calculator, exact match, OfficeQA fuzzy match, LongMemEval type-specific judge.

### `result_repository.py` — `ResultRepository`

Interface for persisting benchmark outputs. Provides methods to save per-trial raw results (`save_trial_results`, `save_ask_trial_results`), per-trial computed metrics (`save_trial_metrics`), and cross-trial aggregations (`save_aggregated_results`).

**Adapters:** JSON file repository.

### `llm_judge.py` — `LLMJudge`

Interface for LLM-based semantic alignment evaluation. `evaluate` returns a simple boolean (correct or not), while `evaluate_with_details` returns a dictionary with vote counts, token usage, and reasoning. Used by `AskMetricsCalculator` adapters that need LLM-based scoring.

**Adapters:** DSPy ensemble voting judge, LongMemEval type-specific judge.

### `database_manager.py` — `DatabaseManager`

Interface for managing database collections. Provides `create_collection` (with schema, vectorizer config, and optional recreate), `batch_insert` (with a transform function mapping items to collection properties), and `close`.

**Adapters:** Weaviate collection manager.

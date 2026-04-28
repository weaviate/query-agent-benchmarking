# Query Agent Benchmarking

A Python library for benchmarking retrieval systems against standard IR datasets. Built for [Weaviate's Query Agent](https://docs.weaviate.io/agents/query), but designed to evaluate any retriever that returns ranked document IDs.

## News 📯

[9/25] 📊 Search Mode Benchmarking is [live](https://weaviate.io/blog/search-mode-benchmarking) on the Weaviate Blog.

## Installation

```bash
pip install query-agent-benchmarking
```

## Quick Start

### Evaluate your own retriever

Bring your own retriever by passing any object that implements the `SearchAgent` protocol:

```python
from query_agent_benchmarking import run_search_eval, SearchAgent, ObjectID

class MyRetriever:
    """Any class with a run() method returning list[ObjectID]."""

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        # Your retrieval logic here
        results = my_search_function(query)
        return [ObjectID(object_id=doc_id) for doc_id in results]

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        return self.run(query, tenant)

    async def initialize_async(self) -> None:
        pass  # Set up async resources (e.g., connection pools)

    async def close_async(self) -> None:
        pass  # Clean up async resources

metrics = run_search_eval(
    search_dataset="beir/scifact/test",
    search_agent=MyRetriever(),
)
```

The library handles dataset loading, query execution, metric computation (Recall@K, nDCG@K, etc.), and results aggregation. See `SearchAgent` in `query_agent_benchmarking/internal/core/ports/search_agent.py` for the full protocol definition.

### Evaluate with custom queries

```python
from query_agent_benchmarking import run_search_eval, InMemoryQuery, DocsCollection

queries = [
    InMemoryQuery(question="What is vector search?", dataset_ids=["doc_1", "doc_5"]),
    InMemoryQuery(question="How does HNSW work?", dataset_ids=["doc_3"]),
]

metrics = run_search_eval(
    docs_collection=DocsCollection(
        collection_name="MyCollection",
        content_key="content",
        id_key="doc_id",
    ),
    queries=queries,
    search_agent=MyRetriever(),
)
```

### Evaluate Weaviate's built-in agents

```python
import query_agent_benchmarking

# Run with a built-in agent
query_agent_benchmarking.run_search_eval(
    search_dataset="beir/scifact/test",
    agent_name="query-agent-search-only",
)

# Compare multiple agents
query_agent_benchmarking.compare_search_agents(
    search_dataset="beir/scifact/test",
    agent_names=["hybrid-search", "query-agent-search-only"],
)
```

## How to Run Scripts 🧰

Populate Weaviate with benchmark data:
```
uv run python3 scripts/populate-db.py
```

Run eval:
```
uv run python3 scripts/run-search-benchmark.py
```

See `query_agent_benchmarking/benchmark-config.yml` to change the dataset populated in your Weaviate instance, as well as ablate `hybrid-search` or `query-agent-search-only`, as well as the number of samples and concurrency parameters.

## Documentation

- [1. Populate Database](docs/1.populate-db.md) — Load benchmark datasets into Weaviate
- [2. Run Built-in Evals](docs/2.run-built-in-evals.md) — Evaluate Weaviate agents on standard benchmarks
- [3. Run Custom Evals](docs/3.run-custom-evals.md) — Bring your own retriever, queries, or collections
- [Experimental](docs/experimental.md) — Synthetic benchmark creation and hard negatives

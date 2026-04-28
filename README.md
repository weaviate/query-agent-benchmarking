# Query Agent Benchmarking

A tool for benchmarking retrieval and question answering systems. Built for [Weaviate's Query Agent](https://docs.weaviate.io/agents/query), but designed to evaluate any system you can plug in.

It supports two evaluation modes:
- **Search** — Ranked retrieval evaluation using IR metrics (Recall@K, nDCG@K, Coverage, alpha-nDCG)
- **Ask** — Question answering evaluation using LLM-as-judge (DSPy-based ensemble voting for semantic alignment) or exact match accuracy

## News 📯

[9/25] 📊 Search Mode Benchmarking is [live](https://weaviate.io/blog/search-mode-benchmarking) on the Weaviate Blog.

## Quick Start

Clone the repo and install dependencies:
```bash
git clone https://github.com/weaviate/query-agent-benchmarking.git
cd query-agent-benchmarking
uv sync
```

Populate Weaviate with benchmark data:
```bash
uv run python3 scripts/populate-db.py
```

Run search eval:
```bash
uv run python3 scripts/run-search-benchmark.py
```

Run ask eval:
```bash
uv run python3 scripts/run-ask-benchmark.py
```

See `query_agent_benchmarking/benchmark-config.yml` to change the dataset, agent type (`hybrid-search`, `query-agent-search-mode`, etc.), number of samples, and concurrency parameters.

## Using as a Python Library

You can also install the package as a dependency and use it programmatically:

```bash
pip install query-agent-benchmarking
```

### Evaluate Weaviate's built-in agents

```python
import query_agent_benchmarking

# Search eval
query_agent_benchmarking.run_search_eval(
    search_dataset="beir/scifact/test",
    agent_name="query-agent-search-mode",
)

# Compare multiple search agents
query_agent_benchmarking.compare_search_agents(
    search_dataset="beir/scifact/test",
    agent_names=["hybrid-search", "query-agent-search-mode"],
)

# Ask eval
query_agent_benchmarking.run_ask_eval(
    ask_dataset="multihoprag",
    agent_name="query-agent-ask-mode",
)
```

### Bring your own retriever

Pass any object that implements the `SearchAgent` protocol directly to `run_search_eval`:

```python
from query_agent_benchmarking import run_search_eval, ObjectID

class MyRetriever:
    """Any class with a run() method returning list[ObjectID]."""

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        results = my_search_function(query)
        return [ObjectID(object_id=doc_id) for doc_id in results]

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        return self.run(query, tenant)

    async def initialize_async(self) -> None: pass
    async def close_async(self) -> None: pass

metrics = run_search_eval(
    search_dataset="beir/scifact/test",
    search_agent=MyRetriever(),
)
```

The library handles dataset loading, query execution, metric computation (Recall@K, nDCG@K, etc.), and results aggregation. See [Bring Your Own Retriever](docs/3.run-custom-evals.md#bring-your-own-retriever) for the full protocol definition and more examples.

### Evaluate with custom questions and answers

```python
from query_agent_benchmarking import run_ask_eval, DocsCollection, InMemoryAskQuery

queries = [
    InMemoryAskQuery(
        question="What is HyDE?",
        ground_truth_answer="HyDE stands for Hypothetical Document Embeddings...",
    ),
]

run_ask_eval(
    docs_collection=DocsCollection(
        collection_name="MyDocs",
        content_key="content",
        id_key="doc_id",
    ),
    queries=queries,
)
```

## Documentation

- [1. Populate Database](docs/1.populate-db.md) — Load benchmark datasets into Weaviate
- [2. Run Built-in Evals](docs/2.run-built-in-evals.md) — Evaluate Weaviate agents on standard search and ask benchmarks
- [3. Run Custom Evals](docs/3.run-custom-evals.md) — Bring your own retriever, QA system, queries, or collections
- [Experimental](docs/experimental.md) — Synthetic benchmark creation and hard negatives

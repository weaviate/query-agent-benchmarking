# Query Agent Benchmarking

This repo contains a package for benchmarking the performance of Weaviate's Query Agent.

## News 📯

[9/25] 📊 Search Mode Benchmarking is live on the [Weaviate Blog](https://weaviate.io/blog/search-mode-benchmarking).

## How to Run 🧰

Populate Weaviate with benchmark data:
```
uv run python3 scripts/populate-db.py
```

Run eval:
```
uv run python3 scripts/run-search-benchmark.py
```

See `query_agent_benchmarking/benchmark-config.yml` to change the dataset populated in your Weaviate instance, as well as ablate `hybrid-search` or `query-agent-search-only`, as well as the number of samples and concurrency parameters.

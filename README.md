# Query Agent Benchmarking

This repo contains a tool for benchmarking the performance of the Weaviate Query Agent.

Populate Weaviate with benchmark data:
```
uv run python benchmarker/populate-db.py
```

Run eval:
```
uv run python benchmarker/benchmark-run.py
```

See `benchmarker/config.yml` to change the dataset populated in your Weaviate instance, as well as ablate `hybrid-search` or `query-agent-search-only`, as well as the number of samples and concurrency parameters.

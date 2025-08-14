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

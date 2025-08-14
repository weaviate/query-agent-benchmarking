# Query Agent Benchmarking

This repo contains a tool for benchmarking the performance of the Weaviate Query Agent.

Run eval with: (See `benchmarker/config.yml` to setup experiment)
```
uv run python benchmarker/benchmark-run.py
```


Populate Weaviate with benchmark data:
```
uv run python benchmarker/populate-db.py
```

# Query Agent Benchmarking

This repo contains a tool for benchmarking the performance of the Weaviate Query Agent.

Run eval with `uv run python benchmarker/benchmark-run.py`. (See `benchmarker/config.yml` to setup experimental details).

Populate Weaviate with benchmark data: `uv run python benchmarker/populate-db.py`.

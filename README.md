# Query Agent Benchmarking

This repo contains a tool for benchmarking the performance of the Weaviate Query Agent end-to-end.

### Test Query Agent in Prod

```bash
python3 benchmarker/src/benchmark-run.py --num-samples 5
```

### Test a locally hosted Query Agent

```bash
python3 benchmarker/src/benchmark-run.py --agents-host http://localhost:8000 --num-samples 5
```

# Supported Datasets

- FreshStack (Angular, Godot, LangChain, Laravel, Yolo)
- EnronQA (dasovich-j)
- WixQA

# Query Agent Prod Results

*Note these are for 10 queries per dataset*

| Dataset Name | Recall | Latency (seconds) | # Searches |
|--------------|--------|-------------------|------------|
| FreshStack Angular | 0.5 | 18.54 | 2.3 |
| FreshStack Godot | 0.38 | 15.32 | 2.6 |
| FreshStack LangChain | 0.5 | 17.85 | 2.0 |
| FreshStack Laravel | 0.35 | 19.07 | 2.6 |
| FreshStack Yolo | 0.26 | 23.7 | 1.9 |

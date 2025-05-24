# Query Agent Benchmarking

This repo contains a tool for benchmarking the performance of the Weaviate Query Agent end-to-end.

### Test Query Agent in Prod

```bash
python3 benchmarker/src/benchmark-run.py --num-samples 5
```

### Test a locally hosted Query Agent

```bash
python3 benchmarker/src/benchmark-run.py --agents-host http://localhost:8000 --num-samples 5 --use-async True
```

# Supported Datasets

- FreshStack LangChain
- EnronQA (dasovich-j)
- WixQA
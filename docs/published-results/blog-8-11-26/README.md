# Scaling Test-Time Compute in Search Mode — Published Results (2026-08-11)

Results backing the blog post **"Scaling Test-Time Compute in Search Mode"** (published 2026-08-11, slug: `search-mode-effort`). The blog compares the Query Agent's Search Mode at three `effort` tiers (`medium`, `high`, `ultrahigh`) against Weaviate's Hybrid Search (BM25 + Snowflake Arctic 2.0 embeddings, RRF fusion) across 8 benchmarks:

- **BRIGHT** (5 subsets): Biology, Earth Science, Economics, Psychology, Robotics
- **IRPAPERS** (text-only transcriptions)
- **WixQA**
- **OBLIQ-Bench** (Congress Hearings subset)

## Layout

```
blog-8-11-26/
├── results/                       # Published runs, one folder per benchmark & system
│   └── <benchmark>/
│       ├── hybrid-search/
│       ├── effort-medium/
│       ├── effort-high/
│       └── effort-ultrahigh/
│           ├── aggregate.json           # Cross-trial mean/std/min/max
│           ├── trial-N.json             # Per-query results for trial N
│           └── trial-N-metrics.json     # Per-trial metrics
```

## Run provenance

The Search Mode effort tiers and the Hybrid Search baseline for a given benchmark share a sweep timestamp:

| Benchmark | Raw run prefix (sweep) |
|---|---|
| bright-biology | `bright-biology-{hybrid-search,query-agent-search-mode}-3-20260719-082823` |
| bright-earth_science | `bright-earth_science-…-3-20260719-154727` |
| bright-economics | `bright-economics-…-3-20260719-171303` |
| bright-psychology | `bright-psychology-…-3-20260719-183616` |
| bright-robotics | `bright-robotics-…-3-20260719-200400` |
| irpapers-text-only | `irpapers-text-only-…-3-20260720-110210` |
| wixqa | `wixqa-…-3-20260723-173813` |
| obliq-bench-congress | `obliq-bench-congress-query-agent-search-mode-3-20260727-123328` (efforts), `obliq-bench-congress-hybrid-search-1-20260727-123328` (hybrid) |

Search Mode effort runs use 3 trials each. Hybrid Search is nearly deterministic, so the blog reports a single run (the BRIGHT/IRPAPERS/WixQA sweeps ran it 3× — the trials are identical).

## Results

Mean ± standard deviation across trials, ×100. Computed from each folder's `aggregate.json`.

### bright-biology

| System | Trials | Success@1 | Recall@5 | Recall@20 | nDCG@10 | Avg. query time (s) |
|---|---|---|---|---|---|---|
| hybrid-search | 3 | 15.5 ± 0.0 | 10.8 ± 0.0 | 19.9 ± 0.0 | 13.0 ± 0.0 | 0.24 ± 0.01 |
| effort-medium | 3 | 46.0 ± 2.0 | 37.8 ± 1.0 | 52.8 ± 2.0 | 43.0 ± 1.7 | 6.30 ± 0.09 |
| effort-high | 3 | 50.2 ± 0.6 | 44.7 ± 0.7 | 67.4 ± 0.1 | 50.6 ± 0.5 | 7.62 ± 0.18 |
| effort-ultrahigh | 3 | 60.5 ± 1.1 | 51.1 ± 1.0 | 66.3 ± 0.1 | 57.5 ± 0.7 | 9.48 ± 0.44 |

### bright-earth_science

| System | Trials | Success@1 | Recall@5 | Recall@20 | nDCG@10 | Avg. query time (s) |
|---|---|---|---|---|---|---|
| hybrid-search | 3 | 19.0 ± 0.9 | 21.7 ± 0.3 | 33.5 ± 0.8 | 22.9 ± 0.3 | 0.23 ± 0.00 |
| effort-medium | 3 | 56.9 ± 1.5 | 42.7 ± 1.2 | 53.4 ± 0.6 | 47.0 ± 0.7 | 7.25 ± 0.35 |
| effort-high | 3 | 58.3 ± 0.5 | 46.4 ± 0.5 | 60.2 ± 0.6 | 51.6 ± 0.4 | 12.42 ± 0.07 |
| effort-ultrahigh | 3 | 64.4 ± 1.3 | 46.0 ± 1.1 | 59.1 ± 0.7 | 53.3 ± 0.3 | 13.55 ± 0.12 |

### bright-economics

| System | Trials | Success@1 | Recall@5 | Recall@20 | nDCG@10 | Avg. query time (s) |
|---|---|---|---|---|---|---|
| hybrid-search | 3 | 10.7 ± 0.0 | 14.2 ± 0.0 | 26.0 ± 0.0 | 16.5 ± 0.0 | 0.24 ± 0.00 |
| effort-medium | 3 | 26.5 ± 2.8 | 21.9 ± 0.9 | 36.5 ± 2.2 | 27.5 ± 0.7 | 6.45 ± 0.13 |
| effort-high | 3 | 20.4 ± 0.0 | 21.6 ± 1.4 | 32.1 ± 1.5 | 24.0 ± 1.5 | 8.68 ± 0.03 |
| effort-ultrahigh | 3 | 34.0 ± 1.7 | 26.5 ± 1.0 | 33.4 ± 1.0 | 31.3 ± 1.1 | 10.00 ± 0.07 |

### bright-psychology

| System | Trials | Success@1 | Recall@5 | Recall@20 | nDCG@10 | Avg. query time (s) |
|---|---|---|---|---|---|---|
| hybrid-search | 3 | 17.8 ± 0.0 | 19.5 ± 0.0 | 32.9 ± 0.0 | 22.2 ± 0.0 | 0.24 ± 0.00 |
| effort-medium | 3 | 38.6 ± 2.0 | 35.3 ± 0.6 | 49.1 ± 1.9 | 40.1 ± 1.2 | 6.82 ± 0.14 |
| effort-high | 3 | 41.6 ± 1.0 | 39.0 ± 0.5 | 58.9 ± 0.5 | 44.8 ± 0.4 | 12.86 ± 0.52 |
| effort-ultrahigh | 3 | 51.8 ± 1.5 | 48.9 ± 1.6 | 58.4 ± 0.7 | 54.3 ± 1.2 | 14.14 ± 0.19 |

### bright-robotics

| System | Trials | Success@1 | Recall@5 | Recall@20 | nDCG@10 | Avg. query time (s) |
|---|---|---|---|---|---|---|
| hybrid-search | 3 | 12.9 ± 0.0 | 12.9 ± 0.0 | 23.3 ± 0.0 | 14.5 ± 0.0 | 2.26 ± 0.11 |
| effort-medium | 3 | 24.1 ± 0.6 | 24.2 ± 0.9 | 34.3 ± 1.3 | 25.2 ± 0.5 | 8.02 ± 0.16 |
| effort-high | 3 | 25.7 ± 1.0 | 27.7 ± 0.6 | 43.4 ± 0.8 | 30.5 ± 0.6 | 9.53 ± 0.35 |
| effort-ultrahigh | 3 | 45.9 ± 1.5 | 37.6 ± 0.3 | 43.0 ± 1.5 | 41.7 ± 0.9 | 11.56 ± 0.23 |

### irpapers-text-only

| System | Trials | Success@1 | Recall@5 | Recall@20 | Avg. query time (s) |
|---|---|---|---|---|---|
| hybrid-search | 3 | 47.8 ± 0.0 | 76.1 ± 0.0 | 91.1 ± 0.0 | 5.11 ± 0.01 |
| effort-medium | 3 | 58.5 ± 0.3 | 82.6 ± 0.8 | 91.5 ± 0.3 | 11.51 ± 0.12 |
| effort-high | 3 | 61.7 ± 0.0 | 87.6 ± 0.3 | 94.1 ± 0.3 | 12.84 ± 0.25 |
| effort-ultrahigh | 3 | 61.9 ± 0.8 | 92.6 ± 0.6 | 94.3 ± 0.3 | 14.56 ± 0.22 |

### wixqa

| System | Trials | Success@1 | Recall@5 | Recall@20 | Avg. query time (s) |
|---|---|---|---|---|---|
| hybrid-search | 3 | 41.0 ± 0.0 | 74.1 ± 0.0 | 90.4 ± 0.0 | 0.43 ± 0.08 |
| effort-medium | 3 | 66.0 ± 0.9 | 86.2 ± 0.6 | 95.8 ± 0.6 | 6.87 ± 0.12 |
| effort-high | 3 | 66.3 ± 0.3 | 85.6 ± 0.3 | 97.5 ± 0.1 | 10.90 ± 0.05 |
| effort-ultrahigh | 3 | 68.7 ± 1.0 | 88.9 ± 0.7 | 97.7 ± 0.4 | 12.32 ± 0.19 |

### obliq-bench-congress

| System | Trials | Success@1 | Recall@5 | Recall@20 | nDCG@10 | Avg. query time (s) |
|---|---|---|---|---|---|---|
| hybrid-search | 1 | 3.5 | 6.3 | 7.9 | 5.3 | 0.32 |
| effort-medium | 3 | 15.2 ± 0.6 | 17.3 ± 1.2 | 18.8 ± 1.0 | 16.3 ± 0.9 | 12.31 ± 0.37 |
| effort-high | 3 | 17.8 ± 0.8 | 20.2 ± 0.9 | 24.1 ± 0.6 | 19.7 ± 0.9 | 18.42 ± 0.44 |
| effort-ultrahigh | 2 | 24.4 ± 0.0 | 25.4 ± 0.8 | 26.2 ± 0.3 | 25.1 ± 0.4 | 18.66 ± 0.55 |

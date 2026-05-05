# Query Agent Benchmarking Console

A Next.js dashboard for running and visualizing Weaviate Query Agent benchmarks. Supports populating databases (Weaviate and Engram), running search/ask benchmarks, and comparing results.

## Getting Started

1. Start the Python backend server:

```bash
uv run uvicorn query_agent_benchmarking.cmd.server:app --reload
```

2. Start the console dev server:

```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000).

The backend runs on `http://localhost:8000` by default. The console proxies API requests to it via the `BACKEND_URL` environment variable (defaults to `http://localhost:8000`).

## Pages

### Home (`/`)

Central hub with navigation to the three main workflows: Populate, Benchmark, and Results.

### Populate Database (`/populate`)

Load datasets into Weaviate or Engram.

- **Weaviate target**: Configure collection tag, embedding models, MUVERA+HNSW parameters, and recreate options.
- **Engram target**: Ingest LongMemEval conversation sessions into Engram's memory system. After ingestion completes, the page displays a manifest summary showing per-tenant stats (sessions, memories created/updated/deleted) and an expandable table of individual run records mapping each `run_id` back to its source `tenant_id`, `session_id`, and `session_date`.
- **LongMemEval subset**: For LongMemEval datasets, optionally filter tenants by sorted index range `[start, end)`.

Manifests are persisted as JSON files in `results/engram-ingest-*.json` for later reference.

### Engram Run History (`/populate/engram-runs`)

Browse past Engram ingestion runs. Each card shows the dataset, timestamp, session/tenant counts, and total memories created. Click a card to view the full detail page with per-tenant stats and a scrollable, filterable table of all run records.

### Run Benchmark (`/benchmark`)

Execute search or ask benchmarks with configurable parameters (dataset, agent, trials, concurrency, etc.).

### Results (`/results`)

Browse benchmark results grouped by dataset or in a table view. Supports:

- Inline label editing for experiments
- Multi-select comparison across experiments
- Per-trial drill-down with query-level detail
- Experiment deletion

## Data Storage

All result and manifest files are stored in `console/results/`:

- `*-results.json` — aggregated benchmark metrics
- `*-trial-N.json` — per-trial query results
- `*-trial-N-metrics.json` — per-trial performance metrics
- `engram-ingest-*.json` — Engram ingestion run manifests
- `_labels.json` — user-assigned experiment labels

## API Routes

| Route | Method | Description |
|---|---|---|
| `/api/backend/[...path]` | GET/POST | Proxy to Python backend (10-min timeout) |
| `/api/experiments` | GET | List all benchmark experiments |
| `/api/experiments/[id]` | GET/PATCH/DELETE | Single experiment CRUD |
| `/api/engram-runs` | GET | List Engram ingestion manifests |
| `/api/engram-runs/[id]` | GET | Single Engram manifest detail |
| `/api/compare` | GET | Compare experiments |
| `/api/trial` | GET | Trial data |

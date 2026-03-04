# Add LongMemEval

Add the LongMemEval-S benchmark to `query-agent-benchmarking` as a **search evaluation**.

## Dataset

Available on HuggingFace as two subsets:

```python
from datasets import load_dataset

queries = load_dataset("weaviate/longmemeval-s-cleaned", "queries")  # 500 questions
docs = load_dataset("weaviate/longmemeval-s-cleaned", "docs")        # 23,867 sessions
```

### Queries schema

| Field | Type | Description |
|---|---|---|
| `question_id` | `str` | Unique question identifier |
| `question` | `str` | The question text |
| `question_date` | `str` | When the question was asked |
| `answer` | `str` | Ground truth answer (string, some are numeric like "3") |
| `answer_session_ids` | `list[str]` | Ground truth — session IDs that contain the answer |
| `tenant_id` | `str` | Same as `question_id`, maps query to its tenant |

### Docs schema

| Field | Type | Description |
|---|---|---|
| `tenant_id` | `str` | Maps doc to its tenant (= `question_id`) |
| `session_id` | `str` | Unique session identifier (used as doc ID for retrieval) |
| `session_date` | `str` | Timestamp of the session |
| `session_text` | `str` | Full multi-turn conversation concatenated as `role: content` lines |

### Data model

- Each of the 500 questions has its own haystack of ~48 sessions (avg).
- One doc = one concatenated chat session.
- Ground truth labels are at the session level — `answer_session_ids` points to the session(s) that contain the answer.
- `tenant_id` links each query to its haystack of docs.

## Multi-Tenant Weaviate Collection

Each question gets its own **tenant** so the agent searches only that question's haystack.

### Collection creation

```python
from weaviate.classes.config import Configure

client.collections.create(
    name="LongmemevalS",
    multi_tenancy_config=Configure.multi_tenancy(
        enabled=True,
        auto_tenant_creation=True,
    ),
    # properties: session_id, session_date, session_text
    # vectorizer on session_text
)
```

Note: Tenant names must be alphanumeric, underscores, or hyphens, 4–64 characters. The `question_id` values (e.g., `e47becba`) satisfy this.

### Data ingestion

For each question (tenant), insert that question's docs into the tenant:

```
for each unique tenant_id:
    1. filter docs to that tenant
    2. batch-insert into the tenant's partition
```

With `auto_tenant_creation=True`, tenants are created automatically on first insert.

## Evaluation

### Mode: Search

This is a **search evaluation**. For each question:

1. Set the active tenant to the question's `tenant_id`
2. Run the question as a retrieval query against that tenant's sessions
3. Compare retrieved `session_id`s to `answer_session_ids`

### Metrics

- Recall@1, Recall@5, Recall@10
- nDCG@10

### Ground truth mapping

- Retrieved doc ID = `session_id`
- Ground truth IDs = `answer_session_ids`

## Implementation Checklist

### Database layer
- [ ] Extend `database_loader.py` to support multi-tenant collection creation
- [ ] Add LongMemEval entry to the dataset registry with properties: `session_id`, `session_date`, `session_text`
- [ ] Implement per-tenant batch insert

### Dataset loader
- [ ] Add `longmemeval-s` loader to `dataset.py` that fetches from `weaviate/longmemeval-s-cleaned`
- [ ] Map docs to the existing ingestion format (keyed by `tenant_id`)
- [ ] Map queries to `InMemoryQuery` with `dataset_ids = answer_session_ids`

### Agent layer
- [ ] Add tenant-awareness to the agent so it queries within the correct tenant per question

### Benchmark runner
- [ ] Add `longmemeval-s` to `DATASET_METRICS` in `query_agent_benchmark.py` (Recall@1/5/20, nDCG@10)
- [ ] Add to `supported_search_datasets` in config

### Future work (out of scope for now)
- Ask evaluation using `answer` field with LLM-as-judge
- Temporal filtering experiments using `question_date` / `session_date`

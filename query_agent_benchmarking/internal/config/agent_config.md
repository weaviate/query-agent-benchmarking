# Agent Configuration

The benchmarking framework uses agent name strings to select retrieval strategy and (optionally) named-vector targeting.
Set these through `search_agent_name` / `ask_agent_name` in `benchmark-config.yml` or programmatically.

For search benchmarks, use `search_target` for target-vector selection.
Use canonical vector names (for example `text_content_weaviate`, `image_content_weaviate`).
The registry of valid names lives in [`/Users/cshorten/Desktop/query-agent-benchmarking/query_agent_benchmarking/named_vectors.md`](/Users/cshorten/Desktop/query-agent-benchmarking/query_agent_benchmarking/named_vectors.md).

## Search Target Keys

Search target selection precedence is:

1. `search_target` (preferred)
2. `search_target_vector` (legacy)
3. `target_vector` (legacy)

If `search_agent_name` already includes an inline suffix (for example `hybrid-search[text_content_weaviate]`), do not also set `search_target`/`search_target_vector`.
The runner raises an error when both are provided.

For provider headers, prefer:
- `embedding_providers: "auto"` (infers providers from selected target vectors), or
- explicit providers (for example `embedding_providers: ["cohere", "voyageai"]`).

Validation uses the named-vector registry in `query_agent_benchmarking.config` and is dataset-aware:

- Built-in datasets: registry lookup key is `search_dataset` (for example `irpapers`)
- Custom collections: registry lookup key is the docs collection name
- If no registry entry matches, the value is passed through unchanged

## Search Agent Names

### `query-agent-search-only`

Wraps the Weaviate [Query Agent](https://weaviate.io/developers/weaviate/agents/query) in search-only mode.

`query-agent-search-only` does not support target-vector suffixes.

```yaml
search_agent_name: "query-agent-search-only"
```

### `hybrid-search`

Weaviate hybrid search (BM25 + vector). Supports an optional `[target_vector]` suffix to specify which named vector(s) to use.

| Agent name | Behavior |
|---|---|
| `hybrid-search` | Auto-detects: uses the single `text_content_*` vector when exactly one exists |
| `hybrid-search[text_content_weaviate]` | BM25 + provider-qualified text vector |
| `hybrid-search[image_content_weaviate]` | BM25 + provider-qualified image vector |
| `hybrid-search[text_content_weaviate+image_content_weaviate]` | BM25 + both vectors combined via `TargetVectors.relative_score` |

```yaml
# Single strategy
search_agent_name: "hybrid-search"
search_target: "text_content_weaviate"
embedding_providers: "auto"
```

```python
# Ablate across strategies with run_search_evals
query_agent_benchmarking.run_search_evals(
    agent_names=[
        "hybrid-search[text_content_weaviate]",
        "hybrid-search[image_content_weaviate]",
        "hybrid-search[text_content_weaviate+image_content_weaviate]",
    ],
)
```

### `vector-search`

Pure vector search via `near_text` (no BM25 component). Useful for isolating embedding quality from keyword matching. Supports the same `[target_vector]` suffix as `hybrid-search`.

| Agent name | Behavior |
|---|---|
| `vector-search` | Auto-detects: uses the single `text_content_*` vector when exactly one exists |
| `vector-search[text_content_weaviate]` | Text embedding only |
| `vector-search[image_content_weaviate]` | Image embedding only |
| `vector-search[text_content_weaviate+image_content_weaviate]` | Both vectors combined via `TargetVectors.relative_score` |

```yaml
search_agent_name: "vector-search"
search_target: "image_content_weaviate"
embedding_providers: "auto"
```

### `external_service`

Bring-your-own-system mode. Sends HTTP POST requests to a configurable host and evaluates the returned results.

```yaml
search_agent_name: "external_service"
external_service_host: "http://localhost:8000/search"
```

Request format:
```json
{"query": "..."}
```

Expected response format:
```json
{"results": ["doc_id_1", "doc_id_2", "..."]}
```

## Ask Agent Names

### `query-agent-ask`

Wraps the Weaviate Query Agent in ask (RAG) mode. The agent retrieves relevant context and generates an answer.

```yaml
ask_agent_name: "query-agent-ask"
```

### `external_service`

Same BYOS pattern as search mode, but for question answering.

```yaml
ask_agent_name: "external_service"
external_service_host: "http://localhost:8000/ask"
```

Request format:
```json
{"question": "...", "oracle_context_id": "..."}
```

Expected response format:
```json
{"answer": "..."}
```

## Multi-Tenancy

Some datasets (e.g. `longmemeval-s`, `longmemeval-m`) use Weaviate multi-tenancy, where each question has its own isolated haystack of documents stored in a separate tenant. Multi-tenancy is handled transparently by the agent layer — no special config is needed.

### How it works

1. **Data loading**: Each `InMemoryQuery` carries an optional `tenant_id` field. For multi-tenant datasets this is set during loading (e.g. from the `tenant_id` column in the HuggingFace dataset).

2. **Query execution**: `run_search_queries` / `run_search_queries_async` in `query_agent_benchmark.py` passes `query.tenant_id` through to the agent:
   ```python
   response = query_agent.run(query.question, tenant=query.tenant_id)
   ```

3. **Agent layer**: `SearchAgentBuilder.run()` and `run_async()` accept an optional `tenant` parameter. When provided, `_get_collection(tenant)` calls `collection.with_tenant(tenant)` to get a tenant-scoped handle before running the query:
   ```python
   def _get_collection(self, tenant=None):
       col = self.weaviate_collection
       if tenant is not None:
           col = col.with_tenant(tenant)
       return col
   ```

4. **Non-multi-tenant datasets**: `tenant_id` defaults to `None`, so the collection is used as-is with no tenant scoping.

### DB population

Multi-tenancy is configured in the `DatasetSpec` via the builder:
```python
(DatasetSpecBuilder("longmemeval-s")
    .with_multi_tenancy(tenant_id_field="tenant_id")
    ...
    .build())
```

This sets `auto_tenant_creation=True` on the collection, so tenants are created automatically during batch insert. Each document's `tenant_id` field determines which tenant it is inserted into.

## Target Vector Syntax

There are two equivalent ways to set target vectors for `hybrid-search` and `vector-search`:

1. `search_target` config key (preferred): `text_content_<provider>`, `image_content_<provider>`, or combinations
2. Explicit agent suffixes: `[text_content_<provider>]`, `[image_content_<provider>]`, `[text_content_<provider>+image_content_<provider>]`

Validation is dataset-aware through the named-vector registry in `query_agent_benchmarking.config`.
Multiple vectors are combined via `TargetVectors.relative_score` with equal weights.
If omitted, backward-compatible auto-detection is used based on collection schema.

You can inspect supported targets for a specific dataset with:

```python
import query_agent_benchmarking as qab

qab.print_named_vector_targets("irpapers")
```

The target vector config is encoded in the agent name string so that result filenames automatically differentiate between strategies (for example `vidore_v3_hr-hybrid-search-text_content_weaviate-results.json` vs `vidore_v3_hr-hybrid-search-image_content_weaviate-results.json`).

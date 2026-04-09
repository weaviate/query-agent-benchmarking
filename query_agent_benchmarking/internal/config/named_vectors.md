# Named Vectors Registry

This file is the human-readable registry for named-vector targets used at query time.

Use these names with:

- `search_target` in `benchmark-config.yml`
- inline suffixes like `hybrid-search[text_content_weaviate]`

Vector names are provider-qualified in the schema layer:
- text vectors: `text_content_<provider>`
- image vectors: `image_content_<provider>`

Examples:
- `text_content_weaviate`
- `text_content_cohere`
- `image_content_weaviate`

## Registered Datasets

| Dataset pattern | Default | Supported named vectors |
|---|---|---|
| `irpapers` | `text_content_<provider>` | `text_content_<provider>[, ...]`, `image_content_<provider>` |
| `vidore_v3_hr` | `text_content_<provider>` | `text_content_<provider>[, ...]`, `image_content_<provider>` |

The concrete provider suffixes come from database config values (for example
`text_embedding_models`, `text_embedding_model`, and `image_embedding_model`).

## Usage Examples

```yaml
search_agent_name: "hybrid-search"
search_target: "text_content_weaviate"
```

```yaml
search_agent_name: "vector-search"
search_target: "image_content_weaviate"
```

```yaml
search_agent_name: "hybrid-search"
search_target: "text_content_weaviate+image_content_weaviate"
```

## Notes

- `search_target` takes canonical vector names only.
- `embedding_providers: "auto"` is the preferred header strategy at benchmark time.
- `search_target_vector` and `target_vector` remain supported as legacy keys.
- If a dataset/collection has no registry entry, values are passed through unchanged.

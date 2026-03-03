# Database Layer (`query_agent_benchmarking/database`)

This package defines how benchmark datasets are mapped into Weaviate collections, including schema, vector configuration, and data loading.

It now supports both:
- single-vector collections (classic text vectorization), and
- named-vector collections (for multimodal datasets like text + image).

## Design Pattern Rationale

The `DatasetSpec -> DatasetSpecBuilder` handoff is primarily a **Builder pattern**:
- `DatasetSpec` (in `spec.py`) is the final runtime contract (treated as read-only by convention after `build()`).
- `DatasetSpecBuilder` (in `property_builder.py`) handles step-by-step construction, branching config logic, and validation before `build()`.

Why this helps:
- schema assembly remains readable in `database_registry.py`,
- provider/named-vector conditionals are centralized in one place,
- runtime code in `database_loader.py` can consume a stable, pre-validated object.

Two secondary patterns are also used:
- **Registry pattern**: `BenchmarkRegistry` + `resolve_spec(...)` choose a spec for a dataset name.
- **Strategy-style callbacks**: `name_fn` and `item_to_props` in `DatasetSpec` encapsulate naming and mapping behavior per dataset family.

## What Each File Does

- `spec.py`
  - Defines `DatasetSpec`, the runtime contract for one dataset family:
    - collection naming
    - properties
    - vector config
    - item-to-properties mapping

- `property_builder.py`
  - `DatasetSpecBuilder` for composing `DatasetSpec` instances.
  - Handles:
    - text properties / blob properties / dataset id mapping
    - provider-aware text vector config (`weaviate`, `cohere`, `voyageai`)
    - provider-dispatch multi2vec config (currently Weaviate-only)
    - MUVERA as a separate concern from provider dispatch
    - explicit image vectorizer helper (`with_image2vec`)
    - provider-qualified named vector composition (e.g., `text_content_weaviate`, `image_content_weaviate`)

- `database_registry.py`
  - Central registry (`BenchmarkRegistry`) mapping dataset patterns to `DatasetSpec`.
  - `resolve_spec(dataset_name)` picks the matching spec.

- `database_loader.py`
  - Runtime loader utilities:
    - create a collection
    - create/update alias
    - batch insert documents
    - create temporary tagged collections for embedding comparisons

- `database_loader_config.yml`
  - Loader-time defaults:
    - `dataset_name`
    - `text_embedding_models` (preferred, allows multiple text vectors)
    - `text_embedding_model` (legacy single-text key)
    - `image_embedding_model`
    - `embedding_providers` (optional explicit header providers)
    - optional MUVERA/HNSW knobs for image multivector configs

## Public vs Private Functions

In `database_loader.py`, functions are intentionally ordered:
1. public API functions at the top
2. private `_` helpers at the bottom

The public functions are:
- `get_vector_config`
- `create_collection_with_vector_config`
- `database_loader`

The private functions are:
- `_drop_and_create_collection`
- `_batch_insert`
- `_resolve_provider_headers`
- `_load_documents`

Private functions use the Python underscore convention because they are internal implementation details and not part of the stable external API.

## End-to-End Flow (`database_loader`)

`database_loader()` does the following:
1. Read `database_loader_config.yml`.
2. Resolve API-key headers for all configured embedding providers.
3. Load documents in-memory for `dataset_name`.
4. Resolve the dataset spec from `BenchmarkRegistry`.
5. Create (or recreate) the tagged collection.
6. Create/update the alias to point at the tagged collection.
7. Batch insert all documents.

This keeps schema definition (`database_registry.py`) separate from runtime behavior (`database_loader.py`).

## NamedVectors Design

Named vectors are defined in `database_registry.py` via `DatasetSpecBuilder`:
- text vector:
  - `.with_text2vec_provider_named(source_properties=[...], base_name="text_content")`
- image vector:
  - `.with_multi2vec(image_field="image", name="image_content", name_by_provider=True)`
  - or `.with_image2vec(image_field="image", name="image_content", name_by_provider=True)` for `img2vec-neural`

Important: vector names used here must match names used by retrieval at query time (for example `text_content_weaviate` and `image_content_weaviate`).

Project convention:
- `.with_multi2vec(...)` is currently restricted to Weaviate's `multi2vec-weaviate`.
- `.with_multi2vec_weaviate(...)` exists as an explicit convenience wrapper.
- `.with_image2vec(...)` is the explicit helper for `img2vec-neural`.
- MUVERA is configured as a separate concern and only applied when the selected
  provider supports it (currently Weaviate), so provider dispatch can be extended
  later without rewriting MUVERA logic.

Strict API surface:
- there is no `.with_multimodal_vectors(...)` convenience wrapper.
- multimodal configuration must be explicit:
  - `.with_text2vec(...)`
  - plus `.with_multi2vec(...)` or `.with_image2vec(...)`

## Config Conventions

Use:
- `text_embedding_models` for one or more text vectorizers (preferred)
- `text_embedding_model` for a single text vectorizer (legacy fallback)
- `image_embedding_model` for image/multimodal vectorization
- `embedding_providers` for explicit header injection (`"auto"` to infer from models)

Legacy single `embedding_model` usage still works in some call paths for backward compatibility, but new code should prefer the explicit text/image keys.

## Temporary Collections for Embedding Ablations

`create_collection_with_vector_config(...)` is used to build tagged temporary collections (for model comparisons).  
It accepts both `client=` and `weaviate_client=` for backward compatibility, but new code should prefer `client=`.

## Adding a New Dataset

1. Add a dataset pattern entry in `BenchmarkRegistry` using `DatasetSpecBuilder`.
2. Define:
   - collection naming (`with_static_name` / `with_prefix_name`)
   - properties and source field mappings
   - vector configuration (single or named)
3. Ensure your dataset loader returns fields expected by the spec mapper.
4. If named vectors are used, choose stable vector names that the retrieval layer can target.

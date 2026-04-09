from __future__ import annotations

from typing import Any, Optional, Sequence

from query_agent_benchmarking.internal.adapters.database import resolve_spec
from query_agent_benchmarking.internal.utils import parse_embedding_model

supported_search_datasets = (
    "beir/fiqa/test",
    "beir/nq",
    "beir/scifact/test",
    "bright/biology",
    "bright/earth_science",
    "bright/economics",
    "bright/psychology",
    "bright/robotics",
    "enron",
    "freshstack-angular",
    "freshstack-godot",
    "freshstack-langchain",
    "freshstack-laravel",
    "freshstack-yolo",
    "irpapers",
    "irpapers-text-only",
    "longmemeval-m",
    "longmemeval-s",
    "lotte/lifestyle/test/forum",
    "lotte/lifestyle/test/search",
    "lotte/recreation/test/forum",
    "lotte/recreation/test/search",
    "reasonir-biology-subset",
    "vidore_v3_hr",
    "wixqa",
)

supported_ask_datasets = (
    "irpapers",  # Uses IRPapers_Default collection
    "longmemeval-s",  # Uses Engram memory (multi-tenant)
    "longmemeval-m",  # Uses Engram memory (multi-tenant)
    "multihoprag",  # Uses MultiHopRAG_Default collection
    "officeqa",  # Uses OfficeQA_Default collection (local PDFs)
)

supported_embedding_models = (
    "cohere/embed-v4.0",
    "voyageai/voyage-3-large",
    "weaviate/Snowflake/snowflake-arctic-embed-m-v1.5",
    "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0",
)

_NAMED_VECTOR_DATASET_SEEDS = (
    "irpapers",
    "vidore_v3_hr",
)

_VECTORIZER_PROVIDER_MAP = {
    "text2vec-weaviate": "weaviate",
    "multi2multivec-weaviate": "weaviate",
    "img2vec-neural": "weaviate",
    "text2vec-cohere": "cohere",
    "text2vec-voyageai": "voyageai",
    "multi2vec-palm": "google",
}

_PROVIDER_ALIASES = {
    "voyage": "voyageai",
    "voyage-ai": "voyageai",
    "gemini": "google",
    "google-gemini": "google",
    "google_gemini": "google",
}


def _matches_pattern(pattern: str, dataset_name: str) -> bool:
    """Check whether a registry pattern matches a dataset name."""
    if pattern.endswith("/*"):
        prefix = pattern[:-2]
        return dataset_name.startswith(prefix + "/")
    if pattern.endswith("-*"):
        prefix = pattern[:-2]
        return dataset_name.startswith(prefix + "-")
    return dataset_name == pattern


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider aliases to canonical provider names."""
    normalized = provider.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _append_unique(items: list[str], value: str) -> None:
    """Append a string to a list if it is not already present."""
    if value and value not in items:
        items.append(value)


def _provider_from_named_vector_config(vector_config: Any) -> Optional[str]:
    """Infer provider from a named vector config object."""
    vectorizer_cfg = getattr(vector_config, "vectorizer", None)
    vectorizer_enum = getattr(vectorizer_cfg, "vectorizer", None)
    vectorizer_name = getattr(vectorizer_enum, "value", None)
    if not isinstance(vectorizer_name, str):
        return None
    provider = _VECTORIZER_PROVIDER_MAP.get(vectorizer_name)
    if provider is None:
        return None
    return _normalize_provider_name(provider)


def _build_named_vector_target_entry(dataset_name: str) -> Optional[dict[str, Any]]:
    """Build a named-vector target entry directly from the dataset spec."""
    try:
        spec = resolve_spec(dataset_name)
    except Exception:
        return None

    if not isinstance(spec.vector_config, list):
        return None

    vectors: list[str] = []
    providers: dict[str, str] = {}
    for vector_cfg in spec.vector_config:
        vector_name = getattr(vector_cfg, "name", None)
        if not vector_name:
            continue
        vectors.append(vector_name)
        provider = _provider_from_named_vector_config(vector_cfg)
        if provider is not None:
            providers[vector_name] = provider

    if not vectors:
        return None

    alias_base = spec.name_fn(dataset_name)
    default = next(
        (vector_name for vector_name in vectors if vector_name.startswith("text_content_")),
        vectors[0],
    )

    return {
        "pattern": dataset_name,
        "aliases": (alias_base, f"{alias_base}_Default"),
        "default": default,
        "vectors": tuple(vectors),
        "providers": providers,
    }


def _build_named_vector_target_registry() -> tuple[dict[str, Any], ...]:
    """Build the named-vector target registry from known dataset specs."""
    entries: list[dict[str, Any]] = []
    for dataset_name in _NAMED_VECTOR_DATASET_SEEDS:
        entry = _build_named_vector_target_entry(dataset_name)
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


# Named-vector target registry containing canonical vector names and provider metadata.
# This registry is derived from the database-layer dataset specs so schema/query naming stays in sync.
named_vector_target_registry: tuple[dict[str, Any], ...] = _build_named_vector_target_registry()


def get_named_vector_target_entry(dataset_name: str) -> Optional[dict[str, Any]]:
    """Return the named-vector registry entry for a dataset or collection, if one exists."""
    for entry in named_vector_target_registry:
        if _matches_pattern(entry["pattern"], dataset_name):
            return entry
        for alias in entry.get("aliases", ()):
            if dataset_name == alias:
                return entry
    return None


def resolve_named_vector_target(
    dataset_name: Optional[str],
    target_vector: Optional[str],
) -> Optional[str]:
    """
    Validate and normalize canonical target vector names.

    Supports single and multi-target values (e.g., "image_content_weaviate" or
    "text_content_weaviate+image_content_weaviate").
    If no dataset-specific registry entry exists, the raw value is returned.
    """
    if target_vector is None:
        return None
    if not isinstance(target_vector, str):
        raise ValueError(
            f"target_vector must be a string; got {type(target_vector).__name__}"
        )
    raw = target_vector.strip()
    if not raw:
        return None
    if dataset_name is None:
        return raw

    entry = get_named_vector_target_entry(dataset_name)
    if entry is None:
        return raw

    vectors: tuple[str, ...] = entry["vectors"]
    available_tokens = sorted(set(vectors))

    resolved: list[str] = []
    for token in raw.split("+"):
        key = token.strip()
        if not key:
            raise ValueError(f"Invalid target vector expression: '{target_vector}'")
        if key not in vectors:
            raise ValueError(
                f"Unknown target vector '{key}' for dataset '{dataset_name}'. "
                f"Available targets: {available_tokens}"
            )
        resolved.append(key)

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate target vectors are not allowed: '{target_vector}'")

    return "+".join(resolved)


def _parse_embedding_providers(
    embedding_providers: Optional[str | Sequence[str]],
) -> Optional[list[str]]:
    """Parse optional embedding_providers config. None means auto."""
    if embedding_providers is None:
        return None

    if isinstance(embedding_providers, str):
        raw = embedding_providers.strip()
        if not raw or raw.lower() == "auto":
            return None
        providers: list[str] = []
        for token in raw.split(","):
            normalized = _normalize_provider_name(token)
            _append_unique(providers, normalized)
        return providers

    if isinstance(embedding_providers, Sequence):
        providers = []
        saw_auto = False
        for value in embedding_providers:
            if not isinstance(value, str):
                raise ValueError(
                    "embedding_providers sequence must contain only strings."
                )
            normalized = _normalize_provider_name(value)
            if normalized == "auto":
                saw_auto = True
                continue
            _append_unique(providers, normalized)
        if saw_auto and providers:
            raise ValueError(
                "embedding_providers cannot mix 'auto' with explicit providers."
            )
        if saw_auto:
            return None
        return providers

    raise ValueError(
        "embedding_providers must be a string, a sequence of strings, or None."
    )


def resolve_embedding_providers(
    dataset_name: Optional[str],
    target_vector: Optional[str],
    embedding_providers: Optional[str | Sequence[str]] = None,
    embedding_models: Optional[Sequence[Optional[str]]] = None,
) -> list[str]:
    """
    Resolve embedding providers for header injection.

    Resolution order:
    1. explicit `embedding_providers` (if provided and not "auto")
    2. dataset named-vector provider mapping inferred from `target_vector`
       (or registry default when no target is supplied)
    3. legacy model-based provider inference from `embedding_models`
    """
    explicit_providers = _parse_embedding_providers(embedding_providers)
    if explicit_providers is not None:
        return explicit_providers

    providers: list[str] = []
    entry = get_named_vector_target_entry(dataset_name) if dataset_name else None
    provider_map = entry.get("providers", {}) if entry else {}

    target_vectors: list[str] = []
    if target_vector:
        target_vectors = [token.strip() for token in target_vector.split("+") if token.strip()]
    elif entry is not None and entry.get("default"):
        target_vectors = [entry["default"]]

    for vector_name in target_vectors:
        provider = provider_map.get(vector_name)
        if provider:
            _append_unique(providers, _normalize_provider_name(provider))

    if embedding_models:
        for embedding_model in embedding_models:
            if not embedding_model:
                continue
            parsed_provider, _ = parse_embedding_model(embedding_model)
            _append_unique(providers, _normalize_provider_name(parsed_provider))

    return providers


def print_supported_datasets():
    """Print all supported search datasets."""
    print("Supported search datasets:")
    for dataset in supported_search_datasets:
        print(f"  - {dataset}")


def print_supported_ask_datasets():
    """Print all supported ask datasets."""
    print("Supported ask datasets:")
    for dataset in supported_ask_datasets:
        print(f"  - {dataset}")


def print_named_vector_targets(dataset_name: str):
    """Print supported named-vector targets for a dataset."""
    entry = get_named_vector_target_entry(dataset_name)
    if entry is None:
        print(f"No named-vector target registry for dataset: {dataset_name}")
        return

    print(f"Named-vector targets for {dataset_name}:")
    print(f"  - default: {entry['default']}")
    print(f"  - canonical: {', '.join(entry['vectors'])}")
    providers = entry.get("providers", {})
    if providers:
        print("  - providers:")
        for vector_name in entry["vectors"]:
            provider = providers.get(vector_name, "unknown")
            print(f"      {vector_name} -> {provider}")

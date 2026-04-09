"""Canonical provider normalization and API header resolution.

Single source of truth for provider name normalization (previously duplicated
in agent/base.py, config.py, and database/database_loader.py).
"""

import os
from typing import Optional, Sequence

_PROVIDER_ALIASES = {
    "voyage": "voyageai",
    "voyage-ai": "voyageai",
    "gemini": "google",
    "google-gemini": "google",
    "google_gemini": "google",
}

_HEADER_MAP = {
    "cohere": ("X-Cohere-Api-Key", "COHERE_API_KEY"),
    "voyageai": ("X-VoyageAI-Api-Key", "VOYAGEAI_API_KEY"),
    "google": ("X-Goog-Api-Key", "GOOGLE_API_KEY"),
}


def normalize_provider_name(provider: str) -> str:
    """Normalize provider aliases to canonical provider names.

    Examples:
        >>> normalize_provider_name("voyage")
        'voyageai'
        >>> normalize_provider_name("google-gemini")
        'google'
        >>> normalize_provider_name("cohere")
        'cohere'
    """
    normalized = provider.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def get_provider_headers(provider: str) -> dict[str, str]:
    """Get API key headers for a third-party embedding provider.

    Args:
        provider: The canonical provider name (e.g., "cohere", "voyageai", "google").

    Returns:
        Dictionary of headers with API keys from environment variables.

    Raises:
        ValueError: If the required API key environment variable is not set.
    """
    if provider not in _HEADER_MAP:
        return {}

    header_name, env_var = _HEADER_MAP[provider]
    api_key = os.getenv(env_var)

    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}'. "
            f"Please set the {env_var} environment variable."
        )

    return {header_name: api_key}


def parse_embedding_model(embedding_model: str) -> tuple[str, str]:
    """Parse embedding model string into provider and model name."""
    if "/" in embedding_model:
        provider, model_name = embedding_model.split("/", 1)
        return provider.lower(), model_name
    return "weaviate", embedding_model


def resolve_headers_for_models(
    embedding_model: Optional[str] = None,
    text_embedding_model: Optional[str] = None,
    image_embedding_model: Optional[str] = None,
    embedding_providers: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Resolve API key headers for all configured embedding providers.

    Args:
        embedding_model: Legacy single embedding model string.
        text_embedding_model: Text embedding model string.
        image_embedding_model: Image embedding model string.
        embedding_providers: Explicit list of provider names.

    Returns:
        Combined headers dictionary for all resolved providers.
    """
    headers: dict[str, str] = {}

    normalized_providers: Sequence[str] = embedding_providers or []
    if isinstance(normalized_providers, str):
        normalized_providers = [normalized_providers]

    if normalized_providers:
        for provider in normalized_providers:
            headers.update(get_provider_headers(normalize_provider_name(provider)))

    models_to_resolve = [text_embedding_model, image_embedding_model]

    if embedding_model and text_embedding_model is None:
        models_to_resolve.insert(0, embedding_model)
    elif embedding_model and embedding_model != text_embedding_model:
        models_to_resolve.append(embedding_model)

    for model in models_to_resolve:
        if not model:
            continue
        provider, _ = parse_embedding_model(model)
        headers.update(get_provider_headers(normalize_provider_name(provider)))

    return headers

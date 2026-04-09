"""Unified Weaviate client factory.

Consolidates all Weaviate connection creation into a single module.
"""

import os
from typing import Optional

import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout


def get_weaviate_client(headers: Optional[dict[str, str]] = None) -> weaviate.WeaviateClient:
    """Create a synchronous Weaviate Cloud connection.

    Args:
        headers: Optional dictionary of headers (e.g., API keys for Cohere, VoyageAI).

    Returns:
        Connected Weaviate client.
    """
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers=headers or {},
    )


def get_async_weaviate_client(
    headers: Optional[dict[str, str]] = None,
    query_timeout: int = 6000,
):
    """Create an async Weaviate Cloud connection (must be awaited to connect).

    Args:
        headers: Optional dictionary of headers.
        query_timeout: Query timeout in seconds.

    Returns:
        Async Weaviate client (call `await client.connect()` to use).
    """
    return weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        headers=headers or {},
        additional_config=AdditionalConfig(
            timeout=Timeout(query=query_timeout)
        ),
    )

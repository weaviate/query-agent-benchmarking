"""Weaviate direct search adapters implementing the SearchAgent protocol.

Provides hybrid search, vector search (near_text), and BM25 search
against Weaviate collections without using the QueryAgent abstraction.
"""

import os
import re
from typing import Any, Optional, Sequence

import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.query import TargetVectors

from query_agent_benchmarking.internal.core.models import ObjectID
from query_agent_benchmarking.internal.adapters.agents.collection_resolver import resolve_collection_info
from query_agent_benchmarking.internal.adapters.clients.provider_headers import (
    resolve_headers_for_models,
)


# ---------------------------------------------------------------------------
# Shared Weaviate collection mixin
# ---------------------------------------------------------------------------

class _WeaviateCollectionMixin:
    """Shared connection and collection management for direct search adapters."""

    def _init_weaviate(
        self,
        dataset_name,
        docs_collection,
        embedding_model,
        text_embedding_model,
        image_embedding_model,
        embedding_providers,
    ):
        info = resolve_collection_info(dataset_name, docs_collection)
        self.collection_name = info["collection"]
        self.id_property = info["id_property"]
        self.uses_named_vectors = info["uses_named_vectors"]
        self.available_named_vectors = info["available_named_vectors"]
        self.dataset_name = dataset_name

        self.headers = resolve_headers_for_models(
            embedding_model=embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
        )
        self._cluster_url = os.getenv("WEAVIATE_URL")
        self._api_key = os.getenv("WEAVIATE_API_KEY")
        self._client: Any = None
        self._collection: Any = None

    def _connect_sync(self) -> weaviate.WeaviateClient:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=self._cluster_url,
            auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
            headers=self.headers,
        )

    def _connect_async(self):
        return weaviate.use_async_with_weaviate_cloud(
            cluster_url=self._cluster_url,
            auth_credentials=Auth.api_key(self._api_key),
            headers=self.headers,
            additional_config=AdditionalConfig(timeout=Timeout(query=6000)),
        )

    def _get_collection(self, tenant: Optional[str] = None):
        col = self._collection
        if tenant is not None:
            col = col.with_tenant(tenant)
        return col

    def _load_named_vector_metadata_sync(self):
        if self.dataset_name is not None or self._collection is None:
            return
        try:
            config = self._collection.config.get()
            vc = getattr(config, "vector_config", None) or {}
            if isinstance(vc, dict):
                self.available_named_vectors = sorted(vc.keys())
                self.uses_named_vectors = bool(self.available_named_vectors)
            else:
                self.available_named_vectors = []
                self.uses_named_vectors = False
        except Exception as exc:
            print(f"Warning: failed to inspect named vectors for '{self.collection_name}': {exc}")

    async def _load_named_vector_metadata_async(self):
        if self.dataset_name is not None or self._collection is None:
            return
        try:
            config = await self._collection.config.get()
            vc = getattr(config, "vector_config", None) or {}
            if isinstance(vc, dict):
                self.available_named_vectors = sorted(vc.keys())
                self.uses_named_vectors = bool(self.available_named_vectors)
            else:
                self.available_named_vectors = []
                self.uses_named_vectors = False
        except Exception as exc:
            print(f"Warning: failed to inspect named vectors for '{self.collection_name}': {exc}")

    def initialize_sync(self):
        self._client = self._connect_sync()
        self._collection = self._client.collections.use(self.collection_name)
        self._load_named_vector_metadata_sync()

    async def initialize_async(self):
        self._client = self._connect_async()
        await self._client.connect()
        self._collection = self._client.collections.use(self.collection_name)
        await self._load_named_vector_metadata_async()

    async def close_async(self):
        if self._client:
            await self._client.close()

    def close_sync(self):
        if self._client:
            self._client.close()


# ---------------------------------------------------------------------------
# Target vector parsing utilities
# ---------------------------------------------------------------------------

def parse_agent_name(raw: str) -> tuple[str, Optional[str]]:
    """Parse agent name and optional ``[target_vector]`` suffix.

    Examples::

        "hybrid-search"                        → ("hybrid-search", None)
        "hybrid-search[text_content_weaviate]" → ("hybrid-search", "text_content_weaviate")
    """
    trimmed = raw.strip()
    if not trimmed:
        raise ValueError("agent_name cannot be empty")
    if "[" not in trimmed:
        return trimmed, None

    match = re.fullmatch(r"([A-Za-z0-9_-]+)\[([^\[\]]+)\]", trimmed)
    if match is None:
        raise ValueError(
            f"Invalid agent_name format: '{raw}'. "
            "Expected format: 'hybrid-search[target_vector]'."
        )
    base, config = match.groups()
    config = config.strip()
    if not config:
        raise ValueError("Target vector config cannot be empty")
    return base, config


def parse_target_vector_names(raw: str) -> list[str]:
    """Parse ``"a+b"`` into ``["a", "b"]`` with validation."""
    names = [n.strip() for n in raw.split("+")]
    if not names or any(not n for n in names):
        raise ValueError(f"Invalid target vector config: '{raw}'")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate target vectors are not allowed: '{raw}'")
    return names


# ---------------------------------------------------------------------------
# Target vector resolution
# ---------------------------------------------------------------------------

class _TargetVectorMixin:
    """Mixin for resolving target_vector kwargs in hybrid/vector search."""

    target_vector_config: Optional[str]
    uses_named_vectors: bool
    available_named_vectors: list[str]
    collection_name: str

    def _validate_target_vectors(self, names: list[str]) -> None:
        if not self.uses_named_vectors:
            if self.dataset_name is None and not self.available_named_vectors:
                return
            raise ValueError(
                f"Collection '{self.collection_name}' is single-vector; "
                f"target vector config '{'+'.join(names)}' is invalid."
            )
        if self.available_named_vectors:
            invalid = [n for n in names if n not in self.available_named_vectors]
            if invalid:
                raise ValueError(
                    f"Unknown target vector(s) {invalid} for '{self.collection_name}'. "
                    f"Available: {self.available_named_vectors}"
                )

    def _resolve_target_vector(self):
        if self.target_vector_config is None:
            if not self.uses_named_vectors:
                return None
            preferred = [v for v in self.available_named_vectors if v.startswith("text_content_")]
            if len(preferred) == 1:
                return preferred[0]
            if "content_vector" in self.available_named_vectors:
                return "content_vector"
            if len(self.available_named_vectors) == 1:
                return self.available_named_vectors[0]
            if len(self.available_named_vectors) > 1:
                raise ValueError(
                    f"Collection '{self.collection_name}' has multiple named vectors "
                    f"{self.available_named_vectors}. Specify one explicitly."
                )
            return None

        names = parse_target_vector_names(self.target_vector_config)
        self._validate_target_vectors(names)
        if len(names) == 1:
            return names[0]
        return TargetVectors.relative_score({n: 1.0 for n in names})


# ---------------------------------------------------------------------------
# Concrete search adapters
# ---------------------------------------------------------------------------

class WeaviateHybridSearch(_WeaviateCollectionMixin, _TargetVectorMixin):
    """SearchAgent adapter for Weaviate hybrid search."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        docs_collection=None,
        target_vector_config: Optional[str] = None,
        embedding_model: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        image_embedding_model: Optional[str] = None,
        embedding_providers: Optional[Sequence[str]] = None,
    ):
        self._init_weaviate(
            dataset_name, docs_collection,
            embedding_model, text_embedding_model,
            image_embedding_model, embedding_providers,
        )
        self.target_vector_config = target_vector_config

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        if self._client is None:
            self.initialize_sync()
        col = self._get_collection(tenant)
        kwargs = {"query": query, "limit": 20}
        tv = self._resolve_target_vector()
        if tv is not None:
            kwargs["target_vector"] = tv
        response = col.query.hybrid(**kwargs)
        return [ObjectID(object_id=str(o.properties[self.id_property])) for o in response.objects]

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        col = self._get_collection(tenant)
        kwargs = {"query": query, "limit": 20}
        tv = self._resolve_target_vector()
        if tv is not None:
            kwargs["target_vector"] = tv
        response = await col.query.hybrid(**kwargs)
        return [ObjectID(object_id=str(o.properties[self.id_property])) for o in response.objects]


class WeaviateVectorSearch(_WeaviateCollectionMixin, _TargetVectorMixin):
    """SearchAgent adapter for Weaviate near_text (vector) search."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        docs_collection=None,
        target_vector_config: Optional[str] = None,
        embedding_model: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        image_embedding_model: Optional[str] = None,
        embedding_providers: Optional[Sequence[str]] = None,
    ):
        self._init_weaviate(
            dataset_name, docs_collection,
            embedding_model, text_embedding_model,
            image_embedding_model, embedding_providers,
        )
        self.target_vector_config = target_vector_config

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        if self._client is None:
            self.initialize_sync()
        col = self._get_collection(tenant)
        kwargs = {"query": query, "limit": 20}
        tv = self._resolve_target_vector()
        if tv is not None:
            kwargs["target_vector"] = tv
        response = col.query.near_text(**kwargs)
        return [ObjectID(object_id=str(o.properties[self.id_property])) for o in response.objects]

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        col = self._get_collection(tenant)
        kwargs = {"query": query, "limit": 20}
        tv = self._resolve_target_vector()
        if tv is not None:
            kwargs["target_vector"] = tv
        response = await col.query.near_text(**kwargs)
        return [ObjectID(object_id=str(o.properties[self.id_property])) for o in response.objects]


class WeaviateBM25Search(_WeaviateCollectionMixin):
    """SearchAgent adapter for Weaviate BM25 keyword search."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        docs_collection=None,
        embedding_model: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        image_embedding_model: Optional[str] = None,
        embedding_providers: Optional[Sequence[str]] = None,
    ):
        self._init_weaviate(
            dataset_name, docs_collection,
            embedding_model, text_embedding_model,
            image_embedding_model, embedding_providers,
        )

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        if self._client is None:
            self.initialize_sync()
        col = self._get_collection(tenant)
        response = col.query.bm25(query=query, limit=20)
        return [ObjectID(object_id=str(o.properties[self.id_property])) for o in response.objects]

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        col = self._get_collection(tenant)
        response = await col.query.bm25(query=query, limit=20)
        return [ObjectID(object_id=str(o.properties[self.id_property])) for o in response.objects]

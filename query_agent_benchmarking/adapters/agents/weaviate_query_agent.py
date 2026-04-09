"""Weaviate QueryAgent adapter implementing SearchAgent and AskAgent protocols.

Wraps ``weaviate.agents.query.QueryAgent`` / ``AsyncQueryAgent`` for both
search-only and ask modes.
"""

import os
from typing import Any, Optional, Sequence

import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout
from weaviate.agents.query import QueryAgent, AsyncQueryAgent

from query_agent_benchmarking.domain.models import ObjectID
from query_agent_benchmarking.ports.ask_agent import AskResponse
from query_agent_benchmarking.adapters.agents.collection_resolver import resolve_collection_info
from query_agent_benchmarking.adapters.clients.provider_headers import (
    resolve_headers_for_models,
)


class WeaviateQueryAgentSearch:
    """SearchAgent adapter wrapping Weaviate QueryAgent in search-only mode."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        docs_collection=None,
        agents_host: Optional[str] = None,
        embedding_model: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        image_embedding_model: Optional[str] = None,
        embedding_providers: Optional[Sequence[str]] = None,
    ):
        info = resolve_collection_info(dataset_name, docs_collection)
        self.collection = info["collection"]
        self.id_property = info["id_property"]
        self.agents_host = agents_host or "https://api.agents.weaviate.io"
        self.headers = resolve_headers_for_models(
            embedding_model=embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
        )
        self._cluster_url = os.getenv("WEAVIATE_URL")
        self._api_key = os.getenv("WEAVIATE_API_KEY")
        self._client: Any = None
        self._agent: Any = None

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

    def initialize_sync(self):
        self._client = self._connect_sync()
        self._agent = QueryAgent(
            client=self._client,
            collections=[self.collection],
            agents_host=self.agents_host,
        )

    async def initialize_async(self):
        self._client = self._connect_async()
        await self._client.connect()
        self._agent = AsyncQueryAgent(
            client=self._client,
            collections=[self.collection],
            agents_host=self.agents_host,
        )

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        if self._agent is None:
            self.initialize_sync()
        response = self._agent.search(query, limit=20)
        return [
            ObjectID(object_id=obj.properties[self.id_property])
            for obj in response.search_results.objects
        ]

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        response = await self._agent.search(query, limit=20)
        return [
            ObjectID(object_id=obj.properties[self.id_property])
            for obj in response.search_results.objects
        ]

    async def close_async(self):
        if self._client:
            await self._client.close()

    def close_sync(self):
        if self._client:
            self._client.close()


class WeaviateQueryAgentAsk:
    """AskAgent adapter wrapping Weaviate QueryAgent in ask mode."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        docs_collection=None,
        agents_host: Optional[str] = None,
        embedding_model: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        image_embedding_model: Optional[str] = None,
        embedding_providers: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        info = resolve_collection_info(dataset_name, docs_collection)
        self.collection = info["collection"]
        self.id_property = info["id_property"]
        self.agents_host = agents_host or "https://api.agents.weaviate.io"
        self.system_prompt = system_prompt
        self.headers = resolve_headers_for_models(
            embedding_model=embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
        )
        self._cluster_url = os.getenv("WEAVIATE_URL")
        self._api_key = os.getenv("WEAVIATE_API_KEY")
        self._client: Any = None
        self._agent: Any = None

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

    def _agent_kwargs(self):
        kwargs = dict(
            client=self._client,
            collections=[self.collection],
            agents_host=self.agents_host,
        )
        if self.system_prompt:
            kwargs["system_prompt"] = self.system_prompt
        return kwargs

    def initialize_sync(self):
        self._client = self._connect_sync()
        self._agent = QueryAgent(**self._agent_kwargs())

    async def initialize_async(self):
        self._client = self._connect_async()
        await self._client.connect()
        self._agent = AsyncQueryAgent(**self._agent_kwargs())

    def run(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        if self._agent is None:
            self.initialize_sync()
        response = self._agent.ask(query)
        return AskResponse(final_answer=response.final_answer, raw_response=response)

    async def run_async(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        response = await self._agent.ask(query)
        return AskResponse(final_answer=response.final_answer, raw_response=response)

    async def close_async(self):
        if self._client:
            await self._client.close()

    def close_sync(self):
        if self._client:
            self._client.close()

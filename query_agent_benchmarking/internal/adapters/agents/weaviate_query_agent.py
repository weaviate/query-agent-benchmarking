"""Weaviate QueryAgent adapter implementing SearchAgent and AskAgent protocols.

Wraps ``weaviate.agents.query.QueryAgent`` / ``AsyncQueryAgent`` for both
search-only and ask modes.
"""

import os
from typing import Any, Literal, Optional, Sequence

import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout
from weaviate.agents.query import QueryAgent, AsyncQueryAgent

from query_agent_benchmarking.internal.core.domain.models import (
    AgentSearch,
    ObjectID,
    SearchAgentResponse,
)
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse
from query_agent_benchmarking.internal.adapters.agents.collection_resolver import resolve_collection_info
from query_agent_benchmarking.internal.adapters.clients.provider_headers import (
    resolve_headers_for_models,
)

Filtering = Literal["recall", "precision"]
Effort = Literal["medium", "high", "ultrahigh"]


def _validate_filtering(filtering: Optional[str]) -> Filtering:
    """Normalize and validate the search filtering strategy.

    "recall" (default) generates multiple Weaviate queries spanning different
    filters/interpretations; "precision" generates a single query targeting the
    most likely interpretation. Defaults to "recall" when unset.
    """
    if filtering is None:
        return "recall"
    if filtering not in ("recall", "precision"):
        raise ValueError(
            f"filtering must be 'recall' or 'precision'; got {filtering!r}."
        )
    return filtering


def _validate_effort(effort: Optional[str]) -> Optional[Effort]:
    """Normalize and validate the search-mode compute effort level.

    "medium" | "high" | "ultrahigh" controls how much compute search mode spends
    on a query. Returns ``None`` when unset, in which case ``effort`` is omitted from
    the request entirely and the agents server applies its own default.
    """
    if effort is None:
        return None
    if effort not in ("medium", "high", "ultrahigh"):
        raise ValueError(
            f"effort must be 'medium', 'high', or 'ultrahigh'; got {effort!r}."
        )
    return effort


def _dump_optional(value: Any) -> Optional[Any]:
    """Dump a (possibly None) pydantic model to a JSON-serializable value.

    The SDK's filter/sort structures are nested pydantic models; we store them
    as plain JSON so the domain stays SDK-agnostic and they persist cleanly.
    """
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _extract_searches(response: Any) -> list[AgentSearch]:
    """Convert a SearchModeResponse's ``searches`` into domain ``AgentSearch``.

    Defensive: the SDK field is optional and may be ``None``; older/other
    response shapes simply yield an empty list.
    """
    raw_searches = getattr(response, "searches", None) or []
    searches: list[AgentSearch] = []
    for s in raw_searches:
        uuid_value = getattr(s, "uuid_value", None)
        searches.append(
            AgentSearch(
                collection=getattr(s, "collection", ""),
                query=getattr(s, "query", None),
                filters=_dump_optional(getattr(s, "filters", None)),
                sort_property=_dump_optional(getattr(s, "sort_property", None)),
                uuid_value=str(uuid_value) if uuid_value is not None else None,
            )
        )
    return searches


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
        filtering: Optional[str] = None,
        effort: Optional[str] = None,
    ):
        info = resolve_collection_info(dataset_name, docs_collection)
        self.collection = info["collection"]
        self.id_property = info["id_property"]
        self.agents_host = agents_host or "https://api.agents.weaviate.io"
        self.filtering: Filtering = _validate_filtering(filtering)
        self.effort: Optional[Effort] = _validate_effort(effort)
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

    def _build_response(self, response: Any) -> SearchAgentResponse:
        """Map a Weaviate ``SearchModeResponse`` to the domain response.

        Captures both the ranked document IDs and the structured ``searches``
        the QueryAgent issued (query text, filters, sort, uuid lookups) so the
        agent's search plan can be persisted and visualized per query.
        """
        retrieved_ids = [
            ObjectID(object_id=obj.properties[self.id_property])
            for obj in response.search_results.objects
        ]
        return SearchAgentResponse(
            retrieved_ids=retrieved_ids,
            searches=_extract_searches(response),
        )

    def _search_kwargs(self) -> dict:
        # `effort` is only forwarded when set: omitting it lets the agents
        # server apply its own default, and keeps clients that don't accept
        # the argument yet working.
        kwargs = {"limit": 20, "filtering": self.filtering}
        if self.effort is not None:
            kwargs["effort"] = self.effort
        return kwargs

    def run(self, query: str, tenant: Optional[str] = None) -> SearchAgentResponse:
        if self._agent is None:
            self.initialize_sync()
        response = self._agent.search(query, **self._search_kwargs())
        return self._build_response(response)

    async def run_async(self, query: str, tenant: Optional[str] = None) -> SearchAgentResponse:
        response = await self._agent.search(query, **self._search_kwargs())
        return self._build_response(response)

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

import re
from typing import Optional, Sequence

import httpx
from weaviate.agents.query import QueryAgent, AsyncQueryAgent
from weaviate.classes.query import TargetVectors

from query_agent_benchmarking.internal.agents.base import BaseAgentBuilder
from query_agent_benchmarking.internal.core.domain.models import ObjectID, DocsCollection


class SearchAgentBuilder(BaseAgentBuilder):
    """
    Agent builder for search mode operations.

    Supports four agent types:
    * `agent_name == "query-agent-search-only"` → Wraps the Weaviate QueryAgent in Search Only Mode.
    * `agent_name == "hybrid-search"` → Wraps Weaviate Hybrid Search. Supports an optional
      `[target_vector]` suffix to specify named vectors, e.g. `"hybrid-search[text_content_weaviate]"`,
      `"hybrid-search[image_content_weaviate]"`, or
      `"hybrid-search[text_content_weaviate+image_content_weaviate]"`.
    * `agent_name == "vector-search"` → Pure vector search via near_text (no BM25). Supports the
      same `[target_vector]` suffix as hybrid-search, e.g. `"vector-search[image_content_weaviate]"`.
    * `agent_name == "bm25-search"` → Pure keyword search via BM25 (no vector component).
    * `agent_name == "external_service"` → Sends requests to an external host for search evaluation.

    The "external_service" mode allows you to bring your own retrieval system
    and use the search infrastructure for evaluation. It sends HTTP POST requests to
    `external_service_host` with `query` and expects back a list of document IDs.

    Expected request format: {"query": "..."}
    Expected response format: {"results": ["doc_id_1", "doc_id_2", ...]}
    """
    SUPPORTED_AGENT_NAMES = {
        "query-agent-search-only",
        "hybrid-search",
        "vector-search",
        "bm25-search",
        "external_service",
    }
    TARGETABLE_AGENT_NAMES = {"hybrid-search", "vector-search"}

    def __init__(
        self,
        agent_name: str,
        dataset_name: Optional[str] = None,
        docs_collection: Optional[DocsCollection] = None,
        agents_host: Optional[str] = None,
        use_async: bool = False,
        embedding_model: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        image_embedding_model: Optional[str] = None,
        external_service_host: Optional[str] = None,
        embedding_providers: Optional[Sequence[str]] = None,
    ):
        parsed_agent_name, parsed_target_vector_config = self._parse_agent_name(agent_name)

        super().__init__(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            use_async=use_async,
            embedding_model=embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
        )

        self.agent_name = parsed_agent_name
        self.target_vector_config = parsed_target_vector_config
        self.external_service_host = external_service_host
        self.weaviate_collection = None

        if self.agent_name not in self.SUPPORTED_AGENT_NAMES:
            raise ValueError(
                f"Unknown agent_name: {self.agent_name}. "
                "Must be 'query-agent-search-only', 'hybrid-search', 'vector-search', 'bm25-search', or 'external_service'"
            )
        if (
            self.target_vector_config is not None
            and self.agent_name not in self.TARGETABLE_AGENT_NAMES
        ):
            raise ValueError(
                "Target vectors are only supported for 'hybrid-search' and 'vector-search'. "
                f"Received: {agent_name}"
            )

        if not use_async:
            self.initialize_sync()

    @staticmethod
    def _parse_agent_name(raw: str) -> tuple[str, Optional[str]]:
        """Parse agent name and optional target vector config from a raw agent name string.

        Examples:
            "hybrid-search"                              → ("hybrid-search", None)
            "hybrid-search[text_content_weaviate]"       → ("hybrid-search", "text_content_weaviate")
            "hybrid-search[text_content_weaviate+image_content_weaviate]"
                                                       → ("hybrid-search", "text_content_weaviate+image_content_weaviate")
            "query-agent-search-only"                    → ("query-agent-search-only", None)
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
                "Expected format: 'hybrid-search[target_vector]' or 'vector-search[target_vector]'."
            )

        base, config = match.groups()
        config = config.strip()
        if not config:
            raise ValueError("Target vector config cannot be empty")
        return base, config

    @staticmethod
    def _parse_target_vector_names(raw: str) -> list[str]:
        """Parse a target vector config string into validated vector names."""
        names = [name.strip() for name in raw.split("+")]
        if not names or any(not name for name in names):
            raise ValueError(f"Invalid target vector config: '{raw}'")
        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate target vectors are not allowed: '{raw}'"
            )
        return names

    def _load_named_vector_metadata_sync(self) -> None:
        """Load named vector metadata for custom collections."""
        if self.dataset_name is not None or self.weaviate_collection is None:
            return
        try:
            collection_config = self.weaviate_collection.config.get()
            vector_config = getattr(collection_config, "vector_config", None) or {}
            if isinstance(vector_config, dict):
                self.available_named_vectors = sorted(vector_config.keys())
                self.uses_named_vectors = bool(self.available_named_vectors)
            else:
                self.available_named_vectors = []
                self.uses_named_vectors = False
        except Exception as exc:
            print(f"Warning: failed to inspect named vectors for '{self.collection}': {exc}")

    async def _load_named_vector_metadata_async(self) -> None:
        """Async variant of named vector metadata loading."""
        if self.dataset_name is not None or self.weaviate_collection is None:
            return
        try:
            collection_config = await self.weaviate_collection.config.get()
            vector_config = getattr(collection_config, "vector_config", None) or {}
            if isinstance(vector_config, dict):
                self.available_named_vectors = sorted(vector_config.keys())
                self.uses_named_vectors = bool(self.available_named_vectors)
            else:
                self.available_named_vectors = []
                self.uses_named_vectors = False
        except Exception as exc:
            print(f"Warning: failed to inspect named vectors for '{self.collection}': {exc}")

    def _validate_target_vectors(self, target_names: list[str]) -> None:
        """Validate target vector names against collection capabilities."""
        if not self.uses_named_vectors:
            # For custom collections, metadata introspection can fail (permissions/network),
            # in which case we allow explicit target vectors to pass through.
            if self.dataset_name is None and not self.available_named_vectors:
                return
            raise ValueError(
                f"Collection '{self.collection}' is single-vector; target vector config "
                f"'{'+'.join(target_names)}' is invalid."
            )

        if self.available_named_vectors:
            invalid = [name for name in target_names if name not in self.available_named_vectors]
            if invalid:
                raise ValueError(
                    f"Unknown target vector(s) {invalid} for collection '{self.collection}'. "
                    f"Available vectors: {self.available_named_vectors}"
                )

    def _resolve_target_vector(self):
        """Resolve the target_vector kwarg for hybrid search.

        Returns None, a string, or a TargetVectors instance depending on config.
        """
        if self.target_vector_config is None:
            if not self.uses_named_vectors:
                return None
            preferred_text_vectors = [
                vector_name
                for vector_name in self.available_named_vectors
                if vector_name.startswith("text_content_")
            ]
            if len(preferred_text_vectors) == 1:
                return preferred_text_vectors[0]
            if "content_vector" in self.available_named_vectors:
                return "content_vector"
            if len(self.available_named_vectors) == 1:
                return self.available_named_vectors[0]
            if len(self.available_named_vectors) > 1:
                raise ValueError(
                    f"Collection '{self.collection}' has multiple named vectors "
                    f"{self.available_named_vectors}. Specify one explicitly, e.g. "
                    "'hybrid-search[text_content_weaviate]'."
                )
            # For custom collections where vector metadata is unavailable, avoid assuming
            # a hardcoded vector name that may not exist.
            return None

        names = self._parse_target_vector_names(self.target_vector_config)
        self._validate_target_vectors(names)

        if len(names) == 1:
            return names[0]
        return TargetVectors.relative_score({name: 1.0 for name in names})

    def initialize_sync(self):
        if self.agent_name == "external_service":
            # External service mode - no Weaviate connection needed
            if not self.external_service_host:
                raise ValueError("external_service_host is required for external_service mode")
            print(f"External service mode initialized with host: {self.external_service_host}")
            return
        
        self.weaviate_client = self._connect_sync()
        
        if self.agent_name == "query-agent-search-only":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=self.agents_host,
            )
        elif self.agent_name in ("hybrid-search", "vector-search"):
            self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
            self._load_named_vector_metadata_sync()
        elif self.agent_name == "bm25-search":
            self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
        else:
            raise ValueError(
                f"Unknown agent_name: {self.agent_name}. "
                "Must be 'query-agent-search-only', 'hybrid-search', 'vector-search', 'bm25-search', or 'external_service'"
            )

    async def initialize_async(self):
        try:
            if self.agent_name == "external_service":
                # External service mode - no Weaviate connection needed
                if not self.external_service_host:
                    raise ValueError("external_service_host is required for external_service mode")
                print(f"External service mode initialized with host: {self.external_service_host}")
                return

            self.weaviate_client = self._connect_async()
            await self.weaviate_client.connect()
            print("Async Weaviate client connected successfully")

            if self.agent_name == "query-agent-search-only":
                self.agent = AsyncQueryAgent(
                    client=self.weaviate_client,
                    collections=[self.collection],
                    agents_host=self.agents_host
                )
                print(f"AsyncQueryAgent initialized for collection: {self.collection}")
                print(f"Using agents host: {self.agents_host}")
            elif self.agent_name in ("hybrid-search", "vector-search"):
                self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
                await self._load_named_vector_metadata_async()
            elif self.agent_name == "bm25-search":
                self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
            else:
                raise ValueError(
                    f"Unknown agent_name: {self.agent_name}. "
                    "Must be 'query-agent-search-only', 'hybrid-search', 'vector-search', 'bm25-search', or 'external_service'"
                )
                
        except Exception as e:
            print(f"Failed to initialize async agent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _get_collection(self, tenant: Optional[str] = None):
        """Return the collection handle, optionally scoped to a tenant."""
        col = self.weaviate_collection
        if tenant is not None:
            col = col.with_tenant(tenant)
        return col

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        """Run synchronous search query."""
        if self.agent_name == "query-agent-search-only":
            response = self.agent.search(query, limit=20)
            results = []
            for obj in response.search_results.objects:
                results.append(ObjectID(object_id=obj.properties[self.id_property]))
            return results

        elif self.agent_name == "hybrid-search":
            col = self._get_collection(tenant)
            target_vector = self._resolve_target_vector()
            hybrid_kwargs = {"query": query, "limit": 20}
            if target_vector is not None:
                hybrid_kwargs["target_vector"] = target_vector
            response = col.query.hybrid(**hybrid_kwargs)
            results = []
            for obj in response.objects:
                results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
            return results

        elif self.agent_name == "vector-search":
            col = self._get_collection(tenant)
            target_vector = self._resolve_target_vector()
            near_text_kwargs = {"query": query, "limit": 20}
            if target_vector is not None:
                near_text_kwargs["target_vector"] = target_vector
            response = col.query.near_text(**near_text_kwargs)
            results = []
            for obj in response.objects:
                results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
            return results

        elif self.agent_name == "bm25-search":
            col = self._get_collection(tenant)
            response = col.query.bm25(query=query, limit=20)
            results = []
            for obj in response.objects:
                results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
            return results

        elif self.agent_name == "external_service":
            # Build request payload
            payload = {"query": query}

            # Send request to external host
            with httpx.Client(timeout=300.0) as client:
                response = client.post(self.external_service_host, json=payload)
                response.raise_for_status()
                data = response.json()

            # Parse results - expect {"results": ["id1", "id2", ...]}
            results = []
            for doc_id in data.get("results", []):
                results.append(ObjectID(object_id=str(doc_id)))
            return results

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        """Run asynchronous search query."""
        try:
            if self.agent_name == "query-agent-search-only":
                response = await self.agent.search(query, limit=20)
                results = []
                for obj in response.search_results.objects:
                    results.append(ObjectID(object_id=obj.properties[self.id_property]))
                return results
            elif self.agent_name == "hybrid-search":
                col = self._get_collection(tenant)
                target_vector = self._resolve_target_vector()
                hybrid_kwargs = {"query": query, "limit": 20}
                if target_vector is not None:
                    hybrid_kwargs["target_vector"] = target_vector
                response = await col.query.hybrid(**hybrid_kwargs)
                results = []
                for obj in response.objects:
                    results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
                return results
            elif self.agent_name == "vector-search":
                col = self._get_collection(tenant)
                target_vector = self._resolve_target_vector()
                near_text_kwargs = {"query": query, "limit": 20}
                if target_vector is not None:
                    near_text_kwargs["target_vector"] = target_vector
                response = await col.query.near_text(**near_text_kwargs)
                results = []
                for obj in response.objects:
                    results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
                return results
            elif self.agent_name == "bm25-search":
                col = self._get_collection(tenant)
                response = await col.query.bm25(query=query, limit=20)
                results = []
                for obj in response.objects:
                    results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
                return results
            elif self.agent_name == "external_service":
                # Build request payload
                payload = {"query": query}

                # Send async request to external host
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(self.external_service_host, json=payload)
                    response.raise_for_status()
                    data = response.json()

                # Parse results - expect {"results": ["id1", "id2", ...]}
                results = []
                for doc_id in data.get("results", []):
                    results.append(ObjectID(object_id=str(doc_id)))
                return results
        except Exception as e:
            print(f"Query '{query[:50]}...' failed with error: {str(e)}")
            raise


# Backwards compatibility alias
AgentBuilder = SearchAgentBuilder

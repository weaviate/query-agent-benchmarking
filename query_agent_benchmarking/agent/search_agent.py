from typing import Optional

import httpx
from weaviate.agents.query import QueryAgent, AsyncQueryAgent

from query_agent_benchmarking.agent.base import BaseAgentBuilder
from query_agent_benchmarking.models import ObjectID, DocsCollection


class SearchAgentBuilder(BaseAgentBuilder):
    """
    Agent builder for search mode operations.
    
    Supports three agent types:
    * `agent_name == "query-agent-search-only"` → Wraps the Weaviate QueryAgent in Search Only Mode.
    * `agent_name == "hybrid-search"` → Wraps Weaviate Hybrid Search.
    * `agent_name == "external_service"` → Sends requests to an external host for search evaluation.
    
    The "external_service" mode allows you to bring your own retrieval system
    and use the search infrastructure for evaluation. It sends HTTP POST requests to
    `external_service_host` with `query` and expects back a list of document IDs.
    
    Expected request format: {"query": "..."}
    Expected response format: {"results": ["doc_id_1", "doc_id_2", ...]}
    """
    
    def __init__(
        self,
        agent_name: str,
        dataset_name: Optional[str] = None,
        docs_collection: Optional[DocsCollection] = None,
        agents_host: Optional[str] = None,
        use_async: bool = False,
        embedding_model: Optional[str] = None,
        external_service_host: Optional[str] = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            use_async=use_async,
            embedding_model=embedding_model,
        )
        
        self.agent_name = agent_name
        self.external_service_host = external_service_host
        self.weaviate_collection = None
        
        if not use_async:
            self.initialize_sync()

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
        elif self.agent_name == "hybrid-search":
            self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
        else:
            raise ValueError(
                f"Unknown agent_name: {self.agent_name}. "
                "Must be 'query-agent-search-only', 'hybrid-search', or 'external_service'"
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
            elif self.agent_name == "hybrid-search":
                self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
            else:
                raise ValueError(
                    f"Unknown agent_name: {self.agent_name}. "
                    "Must be 'query-agent-search-only', 'hybrid-search', or 'external_service'"
                )
                
        except Exception as e:
            print(f"Failed to initialize async agent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def run(self, query: str) -> list[ObjectID]:
        """Run synchronous search query."""
        if self.agent_name == "query-agent-search-only":
            response = self.agent.search(query, limit=20)
            results = []
            for obj in response.search_results.objects:
                results.append(ObjectID(object_id=obj.properties[self.id_property]))
            return results
        
        elif self.agent_name == "hybrid-search":
            response = self.weaviate_collection.query.hybrid(
                query=query,
                limit=20
            )
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

    async def run_async(self, query: str) -> list[ObjectID]:
        """Run asynchronous search query."""
        try:
            if self.agent_name == "query-agent-search-only":
                response = await self.agent.search(query, limit=20)
                results = []
                for obj in response.search_results.objects:
                    results.append(ObjectID(object_id=obj.properties[self.id_property]))
                return results
            elif self.agent_name == "hybrid-search":
                if self.dataset_name and self.dataset_name.startswith("irpapers/images"):
                    response = await self.weaviate_collection.query.near_text(
                        query=query,
                        limit=20
                    )
                else:
                    response = await self.weaviate_collection.query.hybrid(
                        query=query,
                        limit=20
                    )
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


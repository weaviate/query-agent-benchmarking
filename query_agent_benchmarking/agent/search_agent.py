from typing import Optional

from weaviate.agents.query import QueryAgent, AsyncQueryAgent

from query_agent_benchmarking.agent.base import BaseAgentBuilder
from query_agent_benchmarking.models import ObjectID, DocsCollection


class SearchAgentBuilder(BaseAgentBuilder):
    """
    Agent builder for search mode operations.
    
    Supports two agent types:
    * `agent_name == "query-agent-search-only"` → Wraps the Weaviate QueryAgent in Search Only Mode.
    * `agent_name == "hybrid-search"` → Wraps Weaviate Hybrid Search.
    """
    
    def __init__(
        self,
        agent_name: str,
        dataset_name: Optional[str] = None,
        docs_collection: Optional[DocsCollection] = None,
        agents_host: Optional[str] = None,
        use_async: bool = False,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            use_async=use_async,
            embedding_model=embedding_model,
        )
        
        self.agent_name = agent_name
        self.weaviate_collection = None
        
        if not use_async:
            self.initialize_sync()

    def initialize_sync(self):
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
                "Must be 'query-agent-search-only' or 'hybrid-search'"
            )

    async def initialize_async(self):
        try:
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
                    "Must be 'query-agent-search-only' or 'hybrid-search'"
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
        
        if self.agent_name == "hybrid-search":
            response = self.weaviate_collection.query.hybrid(
                query=query,
                limit=20
            )
            results = []
            for obj in response.objects:
                results.append(ObjectID(object_id=str(obj.properties[self.id_property])))
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
        except Exception as e:
            print(f"Query '{query[:50]}...' failed with error: {str(e)}")
            raise


# Backwards compatibility alias
AgentBuilder = SearchAgentBuilder


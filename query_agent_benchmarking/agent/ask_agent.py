from typing import Optional, Any
from dataclasses import dataclass

from weaviate.agents.query import QueryAgent, AsyncQueryAgent

from query_agent_benchmarking.agent.base import BaseAgentBuilder
from query_agent_benchmarking.models import DocsCollection


@dataclass
class AskResponse:
    """Response from an ask query."""
    final_answer: str
    raw_response: Any  # The full response object from the agent


class AskAgentBuilder(BaseAgentBuilder):
    """
    Agent builder for ask mode operations.
    
    Supports two agent types:
    * `agent_name == "query-agent-ask"` → Wraps the Weaviate QueryAgent in Ask Mode.
    * `agent_name == "external"` → Uses external context (oracle_context_id) for RAG.
    
    The "external" mode allows you to bring your own retrieval system and just use
    the ask infrastructure for evaluation.
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
        
        if self.agent_name == "query-agent-ask":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=self.agents_host,
            )
        elif self.agent_name == "external":
            # External mode - just need the collection for context fetching
            self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
        else:
            raise ValueError(
                f"Unknown agent_name: {self.agent_name}. "
                "Must be 'query-agent-ask' or 'external'"
            )

    async def initialize_async(self):
        try:
            self.weaviate_client = self._connect_async()
            await self.weaviate_client.connect()
            print("Async Weaviate client connected successfully")
            
            if self.agent_name == "query-agent-ask":
                self.agent = AsyncQueryAgent(
                    client=self.weaviate_client,
                    collections=[self.collection],
                    agents_host=self.agents_host
                )
                print(f"AsyncQueryAgent (ask mode) initialized for collection: {self.collection}")
                print(f"Using agents host: {self.agents_host}")
            elif self.agent_name == "external":
                # External mode - just need the collection for context fetching
                self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
                print(f"External mode initialized for collection: {self.collection}")
            else:
                raise ValueError(
                    f"Unknown agent_name: {self.agent_name}. "
                    "Must be 'query-agent-ask' or 'external'"
                )
                
        except Exception as e:
            print(f"Failed to initialize async agent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def run(
        self, 
        query: str, 
        oracle_context_id: Optional[str] = None
    ) -> AskResponse:
        """
        Run synchronous ask query.
        
        Args:
            query: The question to ask.
            oracle_context_id: Optional context ID for external mode. 
                               If provided in external mode, fetches this specific context.
        """
        if self.agent_name == "query-agent-ask":
            response = self.agent.ask(query)
            return AskResponse(
                final_answer=response.final_answer,
                raw_response=response
            )
        
        elif self.agent_name == "external":
            if oracle_context_id is None:
                raise ValueError("oracle_context_id is required for external mode")
            
            # Fetch oracle context and use external LLM
            # This is a placeholder - users should extend this for their use case
            from weaviate.classes.query import Filter
            
            response = self.weaviate_collection.query.fetch_objects(
                filters=Filter.by_property(self.id_property).like(oracle_context_id),
                return_properties=[self.target_property_name]
            )
            
            if not response.objects:
                raise ValueError(f"No object found with {self.id_property}={oracle_context_id}")
            
            context = response.objects[0].properties[self.target_property_name]
            
            # For external mode, return the context - user should handle LLM call
            return AskResponse(
                final_answer="[EXTERNAL_MODE] Context fetched. Implement your LLM call.",
                raw_response={"context": context, "oracle_context_id": oracle_context_id}
            )

    async def run_async(
        self, 
        query: str,
        oracle_context_id: Optional[str] = None
    ) -> AskResponse:
        """
        Run asynchronous ask query.
        
        Args:
            query: The question to ask.
            oracle_context_id: Optional context ID for external mode.
        """
        try:
            if self.agent_name == "query-agent-ask":
                response = await self.agent.ask(query)
                return AskResponse(
                    final_answer=response.final_answer,
                    raw_response=response
                )
            
            elif self.agent_name == "external":
                if oracle_context_id is None:
                    raise ValueError("oracle_context_id is required for external mode")
                
                from weaviate.classes.query import Filter
                
                response = await self.weaviate_collection.query.fetch_objects(
                    filters=Filter.by_property(self.id_property).like(oracle_context_id),
                    return_properties=[self.target_property_name]
                )
                
                if not response.objects:
                    raise ValueError(f"No object found with {self.id_property}={oracle_context_id}")
                
                context = response.objects[0].properties[self.target_property_name]
                
                return AskResponse(
                    final_answer="[EXTERNAL_MODE] Context fetched. Implement your LLM call.",
                    raw_response={"context": context, "oracle_context_id": oracle_context_id}
                )
                
        except Exception as e:
            print(f"Ask query '{query[:50]}...' failed with error: {str(e)}")
            raise


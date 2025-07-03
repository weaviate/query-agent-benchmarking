import os
from typing import Optional

import dspy
import weaviate
from weaviate.agents.query import QueryAgent, AsyncQueryAgent
from weaviate.agents.query.classes import QueryAgentCollectionConfig
from weaviate.auth import Auth

from benchmarker.src.dspy_rag.rag_programs import (
    VanillaRAG,
    SearchOnlyRAG,
    SearchOnlyWithQueryWriter,
    SearchOnlyWithFilteredQueryWriter,
    SearchQueryWriter
)
from benchmarker.src.dspy_rag.rag_signatures import DSPyAgentRAGResponse

# Organized in ascending complexity of RAG system
RAG_VARIANTS = {
    "search-only":        SearchOnlyRAG,
    "search-only-with-qw":     SearchOnlyWithQueryWriter,
    "search-only-with-fqw": SearchOnlyWithFilteredQueryWriter,
    "vanilla-rag":            VanillaRAG,
    "query-writer-rag":       SearchQueryWriter
}


class AgentBuilder:
    """
    * `agent_name == "query-agent"`  ➜  wraps Weaviate's hosted QueryAgent.
    * `agent_name in RAG_VARIANTS`   ➜  instantiates one of our RAG variants.
    """

    def __init__(
        self,
        dataset_name: str,
        agent_name: str,
        agents_host: Optional[str] = None,
        use_async: bool = False,
    ):
        self.use_async = use_async
        self.agent = None
        self.weaviate_client = None
        
        self.cluster_url = os.getenv("WEAVIATE_URL")
        self.api_key = os.getenv("WEAVIATE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if dataset_name == "enron":
            self.collection = "EnronEmails"
            self.target_property_name = ""
        elif dataset_name == "wixqa":
            self.collection = "WixKB"
            self.target_property_name = "contents"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("freshstack-"):
            subset = dataset_name.split("-")[1].capitalize()
            self.collection = f"Freshstack{subset}"
            self.target_property_name = "docs_text"
            self.id_property = "dataset_id"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.agent_name = agent_name
        self.agents_host = agents_host or "https://api.agents.weaviate.io"

        # TODO: Separate this into `initialize_sync`
        if not use_async:
            self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
            )
            
            # NOTE [Named Vectors]: Replace collections: list[str] with list[QueryAgentCollectionConfig]
            '''
            class QueryAgentCollectionConfig(BaseModel):
            """A collection configuration for the QueryAgent.

            Attributes:
                name: The name of the collection to query.
                tenant: Tenant name for collections with multi-tenancy enabled.
                view_properties: Optional list of property names the agent has the ability to view
                    for this specific collection.
                target_vector: Optional target vector name(s) for collections with named vectors.
                    Can be a single vector name or a list of vector names.
            """

            name: str
            tenant: Union[str, None] = None
            view_properties: Union[list[str], None] = None
            target_vector: Union[str, list[str], None] = None
            '''
            if agent_name == "query-agent":
                self.agent = QueryAgent(
                    client=self.weaviate_client,
                    collections=[
                        QueryAgentCollectionConfig(
                            name=self.collection,
                            target_vector="snowflake-arctic-embed-l-v2_0"
                        )
                    ],
                    agents_host=self.agents_host,
                )
            elif agent_name in RAG_VARIANTS:
                rag_cls = RAG_VARIANTS[agent_name]
                dspy.configure(
                    lm=dspy.LM("openai/gpt-4.1-mini", api_key=self.openai_api_key, cache=False),
                    track_usage=True,
                )
                self.agent = rag_cls(
                    collection_name=self.collection,
                    target_property_name=self.target_property_name,
                )
            else:
                raise ValueError(f"Unknown agent_name: {agent_name}. Must be 'query-agent' or one of {list(RAG_VARIANTS.keys())}")

    async def initialize_async(self):
        if not self.use_async:
            return
            
        print(f"Initializing async connection to {self.cluster_url}")
        
        try:
            if self.agent_name == "query-agent":
                self.weaviate_client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self.cluster_url,
                    auth_credentials=Auth.api_key(self.api_key),
                )
                
                await self.weaviate_client.connect()
                print("Async Weaviate client connected successfully")
            
            
                self.agent = AsyncQueryAgent(
                    client=self.weaviate_client,
                    collections=[self.collection],
                    agents_host=self.agents_host
                )
                print(f"AsyncQueryAgent initialized for collection: {self.collection}")
                print(f"Using agents host: {self.agents_host}")
                
                print("Testing AsyncQueryAgent with a simple query...")
                test_response = await self.agent.run("What is this collection about?")
                print(f"Test query successful: {test_response.final_answer[:100]}...")
            elif self.agent_name in RAG_VARIANTS:
                # `dspy_rag` keeps the client use encapsulated in the `async_weaviate_search_tool`
                rag_cls = RAG_VARIANTS[self.agent_name]
                dspy.configure(
                    lm=dspy.LM("openai/gpt-4.1-mini", api_key=self.openai_api_key, cache=False),
                    track_usage=True,
                )
                self.agent = rag_cls(
                    collection_name=self.collection,
                    target_property_name=self.target_property_name,
                )
            else:
                raise ValueError(f"Unknown agent_name: {self.agent_name}. Must be 'query-agent' or one of {list(RAG_VARIANTS.keys())}")
                
        except Exception as e:
            print(f"Failed to initialize async agent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def close_async(self):
        if self.use_async and self.weaviate_client:
            try:
                await self.weaviate_client.close()
                print("Async connection closed successfully")
            except Exception as e:
                print(f"Warning: Error closing async connection: {str(e)}")

    def run(self, query: str):
        if self.use_async:
            raise RuntimeError("Use run_async() for async agents")
        
        if self.agent_name == "query-agent":
            return self.agent.run(query)
        
        # For DSPy RAG variants, convert DSPyAgentRAGResponse to AgentRAGResponse
        dspy_response = self.agent.forward(query)
        if isinstance(dspy_response, DSPyAgentRAGResponse):
            return dspy_response.to_agent_rag_response()
        return dspy_response
    
    async def run_async(self, query: str):            
        try:
            if self.agent_name == "query-agent":
                response = await self.agent.run(query)
                return response
            else:
                dspy_response = await self.agent.aforward(query)
                if isinstance(dspy_response, DSPyAgentRAGResponse):
                    return dspy_response.to_agent_rag_response()
                return dspy_response
        except Exception as e:
            print(f"Query '{query[:50]}...' failed with error: {str(e)}")
            raise

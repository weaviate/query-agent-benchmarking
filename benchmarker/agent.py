import os
from typing import Optional

import dspy
import weaviate
from weaviate.agents.query import QueryAgent, AsyncQueryAgent
from weaviate.auth import Auth

class AgentBuilder:
    """
    * `agent_name == "query-agent-search-only"`  ➜  wraps Weaviate's hosted QueryAgent in Search Only Mode.
    * `agent_name == "hybrid-search"` ➜  wraps Weaviate Hybrid Search.
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
            
            if agent_name == "query-agent-search-only":
                self.agent = QueryAgent(
                    client=self.weaviate_client,
                    collections=[self.collection],
                    agents_host=self.agents_host,
                )
            else:
                raise ValueError(f"Unknown agent_name: {agent_name}. Must be 'query-agent-search-only' or 'hybrid-search'")

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
            else:
                raise ValueError(f"Unknown agent_name: {self.agent_name}. Must be 'query-agent-search-only' or 'hybrid-search'")
                
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
        
        if self.agent_name == "query-agent-search-only":
            return self.agent.run(query)
        
        if self.agent_name == "hybrid-search":
            return self.
        return self.agent.forward(query)
    
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

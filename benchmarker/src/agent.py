import os
from typing import Optional

import dspy
import weaviate
from weaviate.agents.query import QueryAgent, AsyncQueryAgent
from weaviate.auth import Auth

from benchmarker.src.dspy_rag import (
    VanillaRAG,
    SearchOnlyRAG,
    SearchOnlyWithQueryWriter,
    SearchQueryWriter
)

RAG_VARIANTS = {
    "vanilla-rag":            VanillaRAG,
    "search-only":        SearchOnlyRAG,
    "search-only-with-qw":     SearchOnlyWithQueryWriter,
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
        elif dataset_name == "freshstack-langchain":
            self.collection = "FreshStackLangChain"
            self.target_property_name = "docs_text"
            self.id_property = "dataset_id"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.agent_name = agent_name
        self.agents_host = agents_host or "https://api.agents.weaviate.io"

        if not use_async:
            self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
            )
            
            if agent_name == "query-agent":
                self.agent = QueryAgent(
                    client=self.weaviate_client,
                    collections=[self.collection],
                    agents_host=self.agents_host,
                )
            elif agent_name in RAG_VARIANTS:
                rag_cls = RAG_VARIANTS[agent_name]
                dspy.configure(
                    lm=dspy.LM("openai/gpt-4o", api_key=self.openai_api_key, cache=False),
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
            self.weaviate_client = weaviate.use_async_with_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=Auth.api_key(self.api_key),
                headers={"X-OpenAI-Api-Key": self.openai_api_key},
            )
            
            await self.weaviate_client.connect()
            print("Async Weaviate client connected successfully")
            
            if self.agent_name == "query-agent":
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
                raise NotImplementedError(f"Async not implemented for {self.agent_name}")
            else:
                raise ValueError(f"Unknown agent_name: {self.agent_name}")
                
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
        return self.agent.forward(query)
    
    async def run_async(self, query: str):
        if not self.use_async:
            raise RuntimeError("Use run() for sync agents")
            
        if self.agent is None:
            raise RuntimeError("Async agent not initialized. Call initialize_async() first.")
            
        if self.weaviate_client is None:
            raise RuntimeError("Async Weaviate client not initialized.")
            
        try:
            if self.agent_name == "query-agent":
                response = await self.agent.run(query)
                return response
            raise NotImplementedError(f"Async not implemented for {self.agent_name}")
        except Exception as e:
            print(f"Query '{query[:50]}...' failed with error: {str(e)}")
            raise

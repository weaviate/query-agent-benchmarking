import os
import asyncio
from typing import Optional

import weaviate
from weaviate.agents.query import QueryAgent, AsyncQueryAgent
from weaviate.auth import Auth
from benchmarker.models import ObjectID

class AgentBuilder:
    """
    * `agent_name == "query-agent-search-only"`  ➜  Wraps the Weaviate QueryAgent in Search Only Mode.
    * `agent_name == "hybrid-search"` ➜  Wraps Weaviate Hybrid Search.
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
            self.target_property_name = "" # I think we can remove this...
        elif dataset_name == "wixqa":
            self.collection = "WixKB"
            self.target_property_name = "contents"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("freshstack-"):
            subset = dataset_name.split("-")[1].capitalize()
            self.collection = f"Freshstack{subset}"
            self.target_property_name = "docs_text"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("beir/"):
            self.collection = f"Beir{dataset_name.split('beir/')[1].replace('-', '_').replace('/', '_').capitalize()}"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("lotte/"):
            self.collection = f"Lotte{dataset_name.split('lotte/')[1].replace('/', '_').capitalize()}"
            self.target_property_name = "text"
            self.id_property = "doc_id"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.agent_name = agent_name
        self.agents_host = agents_host or "https://api.agents.weaviate.io"

        if not use_async:
            self.initialize_sync()
        else:
            self.initialize_async()

    def initialize_sync(self):
        print(f"Initializing sync connection to {self.cluster_url}")
        
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.cluster_url,
            auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
        )
        if self.agent_name == "query-agent-search-only":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=self.agents_host,
            )
        elif self.agent_name == "hybrid-search":
            self.weaviate_collection = self.weaviate_client.collections.get(self.collection)
        else:
            raise ValueError(f"Unknown agent_name: {self.agent_name}. Must be 'query-agent-search-only' or 'hybrid-search'")

    async def initialize_async(self):            
        print(f"Initializing async connection to {self.cluster_url}")
        
        try:
            self.weaviate_client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self.cluster_url,
                    auth_credentials=Auth.api_key(self.api_key),
                )
                
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
                self.weaviate_collection = self.weaviate_client.collections.get(self.collection)
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

    def run(self, query: str) -> list[ObjectID]:
        if self.agent_name == "query-agent-search-only":
            searcher = self.agent.prepare_search(query)
            # TODO: Interface `retrieved_k` instead of hardcoding `20`
            response = searcher.execute(limit=20, offset=0)
            results = []
            for obj in response.search_results.objects:
                results.append(ObjectID(object_id=obj.properties["dataset_id"]))
            return results
        
        if self.agent_name == "hybrid-search":
            response = self.weaviate_collection.query.hybrid(
                query=query,
                limit=20
            )
            results = []
            for obj in response.objects:
                results.append(ObjectID(object_id=str(obj.properties["dataset_id"])))
            return results
        
    async def run_async(self, query: str):
        try:
            if self.agent_name == "query-agent-search-only":
                searcher = self.agent.prepare_search(query)
                response = await searcher.execute(limit=20, offset=0)
                results = []
                for obj in response.search_results.objects:
                    results.append(ObjectID(object_id=obj.properties["dataset_id"]))
                return results
            elif self.agent_name == "hybrid-search":
                response = await self.weaviate_collection.query.hybrid(
                    query=query,
                    limit=20
                )
                results = []
                for obj in response.objects:
                    results.append(ObjectID(object_id=obj.properties["dataset_id"]))
                return results
        except Exception as e:
            print(f"Query '{query[:50]}...' failed with error: {str(e)}")
            raise

async def main():
    agent = AgentBuilder(
        dataset_name="freshstack-langchain",
        agent_name="hybrid-search",
        agents_host="https://dev-agents.labs.weaviate.io",
        use_async=False,
    )
    response = agent.run("What is this collection about?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
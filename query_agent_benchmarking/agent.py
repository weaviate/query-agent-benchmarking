import os
from typing import Optional, Dict

import weaviate
from weaviate.agents.query import QueryAgent, AsyncQueryAgent
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout
from query_agent_benchmarking.models import ObjectID, DocsCollection
from query_agent_benchmarking.utils import pascalize_name, get_provider_headers


def _parse_embedding_model(embedding_model: str) -> tuple[str, str]:
    """Parse embedding model string into provider and model name."""
    if "/" in embedding_model:
        provider, model_name = embedding_model.split("/", 1)
        return provider.lower(), model_name
    return "weaviate", embedding_model

class AgentBuilder:
    """
    * `agent_name == "query-agent-search-only"`  ➜  Wraps the Weaviate QueryAgent in Search Only Mode.
    * `agent_name == "hybrid-search"` ➜  Wraps Weaviate Hybrid Search.
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
        self.use_async = use_async
        self.agent = None
        self.weaviate_client = None
        
        self.cluster_url = os.getenv("WEAVIATE_URL")
        self.api_key = os.getenv("WEAVIATE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Get provider headers for third-party embedding providers
        self.headers: Dict[str, str] = {}
        if embedding_model:
            provider, _ = _parse_embedding_model(embedding_model)
            self.headers = get_provider_headers(provider)
        
        # NOTE: Update this to use `docs_collection` if both `dataset_name` and `docs_collection` are provided.
        # Require either dataset_name or docs_collection, but not both
        if dataset_name and docs_collection:
            raise ValueError("Cannot specify both dataset_name and docs_collection")
        if not dataset_name and not docs_collection:
            raise ValueError("Must specify either dataset_name or docs_collection")
        

        self.dataset_name = dataset_name # for irpapers/images query toggle
        
        # Handle custom DocsCollection
        if docs_collection:
            self.collection = docs_collection.collection_name
            self.target_property_name = docs_collection.content_key
            self.id_property = docs_collection.id_key
        # NOTE: Might need to add `_Default` for the QueryAgent to avoid the use of the alias    
        # NOTE: Or maybe not, this might just work as is.        
        # Handle built-in datasets
        elif dataset_name == "enron":
            self.collection = "EnronEmails"
            self.target_property_name = ""
            self.id_property = "dataset_id"
        elif dataset_name == "wixqa":
            self.collection = "WixKB"
            self.target_property_name = "contents"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("freshstack-"):
            subset = dataset_name.split("-")[1]
            self.collection = f"Freshstack{pascalize_name(subset)}"
            self.target_property_name = "docs_text"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("beir/"):
            subset = dataset_name.split('beir/')[1]
            self.collection = f"Beir{pascalize_name(subset)}"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("lotte/"):
            lotte_subset = dataset_name.split("/")[1]
            self.collection = f"Lotte{pascalize_name(lotte_subset)}"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("bright/"):
            subset = dataset_name.split('/')[1]
            self.collection = f"Bright{pascalize_name(subset)}"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("irpapers/images"):
            self.collection = "IRPapersImages"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("irpapers/text"):
            self.collection = "IRPapersText"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
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
            headers=self.headers,
        )
        if self.agent_name == "query-agent-search-only":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=self.agents_host,
            )
        # NOTE: Change to `use` for Collection Alias
        elif self.agent_name == "hybrid-search":
            self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
        else:
            raise ValueError(f"Unknown agent_name: {self.agent_name}. Must be 'query-agent-search-only' or 'hybrid-search'")

    async def initialize_async(self):            
        print(f"Initializing async connection to {self.cluster_url}")
        
        try:
            self.weaviate_client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self.cluster_url,
                    auth_credentials=Auth.api_key(self.api_key),
                    headers=self.headers,
                    additional_config=AdditionalConfig(
                        timeout=Timeout(query=6000)
                    ),
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
                self.weaviate_collection = self.weaviate_client.collections.use(self.collection)
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
            # TODO: Interface `retrieved_k` instead of hardcoding `20`
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
        
    async def run_async(self, query: str):
        try:
            if self.agent_name == "query-agent-search-only":
                # TODO: Interface `retrieved_k` instead of hardcoding `20`
                response = await self.agent.search(query, limit=20)
                results = []
                for obj in response.search_results.objects:
                    results.append(ObjectID(object_id=obj.properties[self.id_property]))
                return results
            elif self.agent_name == "hybrid-search":
                if self.dataset_name.startswith("irpapers/images"):
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
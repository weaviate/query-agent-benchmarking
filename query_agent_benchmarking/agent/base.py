import os
from typing import Optional
from abc import ABC, abstractmethod

import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout

from query_agent_benchmarking.models import DocsCollection
from query_agent_benchmarking.database.database_registry import resolve_spec
from query_agent_benchmarking.utils import get_provider_headers, parse_embedding_model


class BaseAgentBuilder(ABC):
    """
    Base class for agent builders that handles common Weaviate connection logic
    and dataset-to-collection mapping.
    """
    
    def __init__(
        self,
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
        self.headers: dict[str, str] = {}
        if embedding_model:
            provider, _ = parse_embedding_model(embedding_model)
            self.headers = get_provider_headers(provider)
        
        # Require either dataset_name or docs_collection, but not both
        if dataset_name and docs_collection:
            raise ValueError("Cannot specify both dataset_name and docs_collection")
        if not dataset_name and not docs_collection:
            raise ValueError("Must specify either dataset_name or docs_collection")
        
        self.dataset_name = dataset_name
        
        # Handle custom DocsCollection
        if docs_collection:
            self.collection = docs_collection.collection_name
            self.id_property = docs_collection.id_key
        else:
            spec = resolve_spec(dataset_name)
            self.collection = f"{spec.name_fn(dataset_name)}_Default"
            self.id_property = "dataset_id"

        self.agents_host = agents_host or "https://api.agents.weaviate.io"

    def _connect_sync(self) -> weaviate.WeaviateClient:
        """Create synchronous Weaviate connection."""
        print(f"Initializing sync connection to {self.cluster_url}")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=self.cluster_url,
            auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
            headers=self.headers,
        )

    def _connect_async(self):
        """Create async Weaviate connection (returns client, must be awaited to connect)."""
        print(f"Initializing async connection to {self.cluster_url}")
        return weaviate.use_async_with_weaviate_cloud(
            cluster_url=self.cluster_url,
            auth_credentials=Auth.api_key(self.api_key),
            headers=self.headers,
            additional_config=AdditionalConfig(
                timeout=Timeout(query=6000)
            ),
        )

    @abstractmethod
    def initialize_sync(self):
        """Initialize synchronous agent. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def initialize_async(self):
        """Initialize asynchronous agent. Must be implemented by subclasses."""
        pass

    async def close_async(self):
        """Close async connection."""
        if self.use_async and self.weaviate_client:
            try:
                await self.weaviate_client.close()
                print("Async connection closed successfully")
            except Exception as e:
                print(f"Warning: Error closing async connection: {str(e)}")

    def close_sync(self):
        """Close sync connection."""
        if not self.use_async and self.weaviate_client:
            try:
                self.weaviate_client.close()
                print("Sync connection closed successfully")
            except Exception as e:
                print(f"Warning: Error closing sync connection: {str(e)}")


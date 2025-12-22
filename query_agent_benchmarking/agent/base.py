import os
from typing import Optional, Dict
from abc import ABC, abstractmethod

import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout

from query_agent_benchmarking.models import DocsCollection
from query_agent_benchmarking.utils import pascalize_name, get_provider_headers, parse_embedding_model


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
        self.headers: Dict[str, str] = {}
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
            self.target_property_name = docs_collection.content_key
            self.id_property = docs_collection.id_key
        else:
            self._setup_builtin_dataset(dataset_name)

        self.agents_host = agents_host or "https://api.agents.weaviate.io"

    def _setup_builtin_dataset(self, dataset_name: str):
        """Configure collection settings for built-in datasets."""
        if dataset_name == "enron":
            self.collection = "EnronEmails_Default"
            self.target_property_name = ""
            self.id_property = "dataset_id"
        elif dataset_name == "wixqa":
            self.collection = "WixKB_Default"
            self.target_property_name = "contents"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("freshstack-"):
            subset = dataset_name.split("-")[1]
            self.collection = f"Freshstack{pascalize_name(subset)}_Default"
            self.target_property_name = "docs_text"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("beir/"):
            subset = dataset_name.split('beir/')[1]
            self.collection = f"Beir{pascalize_name(subset)}_Default"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("lotte/"):
            lotte_subset = dataset_name.split("/")[1]
            self.collection = f"Lotte{pascalize_name(lotte_subset)}_Default"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("bright/"):
            subset = dataset_name.split('/')[1]
            self.collection = f"Bright{pascalize_name(subset)}_Default"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("irpapers/images"):
            self.collection = "IRPapersImages_Default"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        elif dataset_name.startswith("irpapers/text"):
            self.collection = "IRPapersText_Default"
            self.target_property_name = "content"
            self.id_property = "dataset_id"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

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


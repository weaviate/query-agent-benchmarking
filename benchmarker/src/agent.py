import os
from typing import Any, Optional

import weaviate
from weaviate.agents.query import QueryAgent

from benchmarker.src.dspy_rag import VanillRAG

class AgentBuilder():
    def __init__(
            self, 
            dataset_name: str,
            agent_name: str,
            agents_host: Optional[str] | None = None,
        ):
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )

        self.agent_name = agent_name

        if dataset_name == "enron":
            self.collections = ["EnronEmails"]
        if dataset_name == "wixqa":
            self.collections = ["WixKB"]

        if agent_name == "query-agent":
            self.agents_host = agents_host or "https://api.agents.weaviate.io"

            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=self.collections,
                agents_host=self.agents_host,
            )
        elif agent_name == "vanilla-rag":
            self.agent = VanillaRAG()
    
    def run(self, query: str):
        response = self.agent.run(query)
        return response


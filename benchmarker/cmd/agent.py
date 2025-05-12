import weaviate
from typing import Any, Optional
from weaviate.agents.query import QueryAgent

class QueryAgentBuilder():
    def __init__(
            self, 
            weaviate_client: weaviate.WeaviateClient,
            dataset_name: str,
            agents_host: Optional[str] | None = None,
        ):
        self.weaviate_client = weaviate_client

        if dataset_name == "enron":
            self.collections = ["EnronEmails"]

        self.agents_host = agents_host or "https://api.agents.weaviate.io"

        if self.agent_name == "query-agent":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=self.collections,
                agents_host=self.agents_host,
            )
    
    def run(self, inputs: dict[str, Any]):
        if self.agent_name == "query-agent":
            response = self.agent.run(**inputs)
            return response


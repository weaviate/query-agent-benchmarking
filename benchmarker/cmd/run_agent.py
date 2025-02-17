import weaviate
from typing import Any
from weaviate.agents.query import QueryAgent, QueryAgentResponse

class AgentWrapper():
    def __init__(
            self, 
            weaviate_client: weaviate.WeaviateClient,
            collections: list[str], 
            config: dict[str, str]
        ):
        self.agent_name = config.agent_name
        self.weaviate_client = weaviate_client

        if self.agent_name == "query-agent":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=collections
            )
    
    def run(self, inputs: dict[str, Any]) -> QueryAgentResponse:
        if self.agent_name == "query-agent":
            response = self.agent.run(**inputs)
            return response


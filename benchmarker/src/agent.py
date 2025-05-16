import os
from typing import Optional

import dspy
import weaviate
from weaviate.agents.query import QueryAgent

from benchmarker.src.dspy_rag import VanillaRAG

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
            self.collection = "EnronEmails"
            self.target_property_name=""
        if dataset_name == "wixqa":
            self.collection = "WixKB"
            self.target_property_name = "contents"
            self.id_property = "dataset_id"

        if agent_name == "query-agent":
            self.agents_host = agents_host or "https://api.agents.weaviate.io"

            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=self.agents_host,
            )
        elif agent_name == "vanilla-rag":
            lm = dspy.LM('openai/gpt-4o', api_key=os.getenv("OPENAI_API_KEY")) # update this to ablate model
            dspy.configure(lm=lm)

            self.agent = VanillaRAG(
                collection_name=self.collection,
                target_property_name=self.target_property_name
            )

    def run(self, query: str):
        if self.agent_name == "query-agent":
            response = self.agent.run(query)
            return response
        if self.agent_name == "vanilla-rag":
            response = self.agent.forward(query)
            return response


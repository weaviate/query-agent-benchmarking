import os
from typing import Optional

import dspy
import weaviate
from weaviate.agents.query import QueryAgent

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
    ):
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )

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

        # ---------------- Hosted QueryAgent ----------------------------------
        if agent_name == "query-agent":
            self.agent = QueryAgent(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=agents_host or "https://api.agents.weaviate.io",
            )
            return

        # ---------------- RAG ablations --------------------------------------
        if agent_name in RAG_VARIANTS:
            rag_cls = RAG_VARIANTS[agent_name]

            # configure DSPy (once)
            dspy.configure(
                lm=dspy.LM("openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), cache=False),
                track_usage=True,
            )

            # instantiate chosen ablation
            self.agent = rag_cls(
                collection_name=self.collection,
                target_property_name=self.target_property_name,
            )
            return

        raise ValueError(f"Unknown agent_name: {agent_name}. Must be 'query-agent' or one of {list(RAG_VARIANTS.keys())}")

    # unified call interface ---------------------------------------------------
    def run(self, query: str):
        if self.agent_name == "query-agent":
            return self.agent.run(query)
        return self.agent.forward(query)

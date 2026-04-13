"""
Engram + DSPy agent for LongMemEval evaluation.

This agent retrieves memories from Engram via hybrid search and answers
questions using DSPy. It is query-only — ingestion is handled separately
by ``database.engram_loader``.
"""

import os
from typing import Optional
from dataclasses import dataclass, field

import dspy
from pydantic import BaseModel

from engram import EngramClient, RetrievalConfig


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class MemoryWithTimestamp(BaseModel):
    memory: str
    time_added: str


class AnswerUserQueryWithMemory(dspy.Signature):
    """Answer the user's question using ONLY facts explicitly stated in the retrieved memories.

    Rules:
    - Do NOT infer, assume, or supplement with outside knowledge.
    - Only mention specific products, brands, or details if they appear verbatim in the memories.
    - If the memories do not contain enough information to fully answer the question, state what IS known from the memories and clearly identify what is missing.
    - Do NOT fabricate plausible-sounding details to fill gaps."""

    user_question: str = dspy.InputField()
    retrieved_memories: list[MemoryWithTimestamp] = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="Answer grounded strictly in the retrieved memories. Only reference specifics (products, names, dates, preferences) that appear explicitly in the memories. State gaps honestly."
    )


# ---------------------------------------------------------------------------
# Response dataclass (matches AskResponse interface)
# ---------------------------------------------------------------------------

@dataclass
class EngramAskResponse:
    final_answer: str
    raw_response: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EngramDSPyAgent:
    """
    Query-only agent that retrieves Engram memories and answers via DSPy.

    Ingestion is handled separately by
    ``query_agent_benchmarking.internal.adapters.database.engram_loader``.

    Usage::

        agent = EngramDSPyAgent(
            engram_base_url="https://dev-engram.labs.weaviate.io",
            dspy_lm_model="openai/gpt-5.4",
            retrieval_limit=10,
        )
        response = agent.run(query="How long did I wait?", tenant_id="001be529")
    """

    def __init__(
        self,
        engram_base_url: str = "https://dev-engram.labs.weaviate.io",
        engram_api_key: Optional[str] = None,
        dspy_lm_model: str = "openai/gpt-5.4",
        retrieval_limit: int = 10,
        retrieval_type: str = "hybrid",
        engram_group: str = "default",
        user_id_prefix: str = "longmemeval-",
    ):
        self.engram_client = EngramClient(
            api_key=engram_api_key or os.environ["ENGRAM_API_KEY"],
            base_url=engram_base_url,
        )
        self.retrieval_limit = retrieval_limit
        self.retrieval_type = retrieval_type
        self.engram_group = engram_group
        self.user_id_prefix = user_id_prefix

        # TODO: dspy.configure() sets a global LM — this will overwrite any
        # existing DSPy LM config in the process. Consider using dspy.context()
        # for scoped configuration once DSPy supports it cleanly.
        lm = dspy.LM(dspy_lm_model, api_key=os.environ.get("OPENAI_API_KEY"))
        dspy.configure(lm=lm)
        self.qa_system = dspy.Predict(AnswerUserQueryWithMemory)

    def run(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        oracle_context_id: Optional[str] = None,
    ) -> EngramAskResponse:
        """
        Retrieve memories from Engram and answer the question using DSPy.

        Args:
            query: The user's question.
            tenant_id: Tenant whose memories to search.
            oracle_context_id: Unused, kept for interface compatibility.
        """
        if tenant_id is None:
            raise ValueError("tenant_id is required for EngramDSPyAgent")

        user_id = f"{self.user_id_prefix}{tenant_id}"

        memories = self.engram_client.memories.search(
            query=query,
            user_id=user_id,
            group=self.engram_group,
            retrieval_config=RetrievalConfig(
                retrieval_type=self.retrieval_type,
                limit=self.retrieval_limit,
            ),
        )

        retrieved = [
            MemoryWithTimestamp(
                memory=m.content,
                time_added=str(m.created_at),
            )
            for m in memories
        ]

        response = self.qa_system(
            user_question=query,
            retrieved_memories=retrieved,
        )

        return EngramAskResponse(
            final_answer=response.answer,
            raw_response={
                "n_memories_retrieved": len(retrieved),
                "memories": [r.model_dump() for r in retrieved],
            },
        )

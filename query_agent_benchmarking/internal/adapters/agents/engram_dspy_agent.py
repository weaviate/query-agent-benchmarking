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

from engram import (
    EngramClient,
    BM25Retrieval,
    FetchRetrieval,
    HybridRetrieval,
    VectorRetrieval,
)

_RETRIEVAL_CLASSES = {
    "hybrid": HybridRetrieval,
    "bm25": BM25Retrieval,
    "vector": VectorRetrieval,
    "fetch": FetchRetrieval,
}


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class MemoryWithTimestamp(BaseModel):
    memory: str
    time_added: str


class AnswerUserQueryWithMemory(dspy.Signature):
    """Answer the user's question using the retrieved memories as your knowledge base.

    Determine the question type, then apply the matching strategy:

    FACTUAL questions ("What is my cat's name?", "Where do I work?", "How many engineers do I lead?"):
    - Answer using ONLY facts explicitly stated in the memories.
    - Only mention specific names, numbers, products, or details if they appear in the memories.
    - Do NOT fabricate or guess facts.

    PREFERENCE / RECOMMENDATION questions ("What should I cook?", "What should I serve?", "What would I enjoy?"):
    - These ask you to USE the user's stored preferences, habits, and interests to make a personalized suggestion.
    - You SHOULD synthesize across memories to form a recommendation grounded in what the user likes, has, or has done.
    - Example: if memories say the user grows basil and cherry tomatoes, and they ask what to cook, suggest dishes featuring those ingredients. This is the correct behavior, not fabrication.
    - Do NOT abstain just because no memory literally answers the question — the memories provide the ingredients for your recommendation.

    BOTH types:
    - Synthesizing facts across multiple memories is expected and encouraged.
    - BEFORE answering, check the question's premises against the memories (see premise_check).
    - If the memories contain NO relevant information at all, say "The information provided is not enough."

    Contradiction vs. gap:
    - A CONTRADICTION is when the question assumes a specific fact (role title, name, date, event, location) and the memories state a DIFFERENT fact. Example: the question says "Software Engineer Manager" but memories say "Senior Software Engineer" — these are two different roles, so this is a contradiction.
    - A GAP is when the memories are simply silent on a topic — they neither confirm nor deny it.
    - If you find a contradiction: you MUST abstain. Say "The information provided is not enough." then explain what the question assumed vs. what the memories actually say.
    - If you find only gaps: answer with whatever relevant information IS available. Do NOT abstain just because the answer is incomplete.

    Temporal reasoning:
    - When memories describe the same thing at different points in time, resolve the timeline. Look for language that distinguishes past states, plans/intentions, and current states:
      * Past: "stored under the bed", "used to", "previously"
      * Plans: "plans to", "intends to", "wants to", "will"
      * Current: present tense statements of fact ("keeps sneakers in a shoe rack", "is in a shoe rack")
    - A current-state memory supersedes a past-state or plan memory about the same topic. If one memory says "sneakers are under the bed" (past) and another says "sneakers are in a shoe rack" (current), the answer is the shoe rack — do NOT hedge or say "unclear."
    - If a memory describes both a past and current state, treat them as distinct facts at different points in time — do not collapse them.
    - When the question asks about the current state ("Where do I currently keep..."), give the most recent state, not the full history.
    - The timestamps on the memories reflect storage time, NOT when the events originally occurred. Reason about temporal order from the content of the memories, not from the timestamps."""

    user_question: str = dspy.InputField()
    retrieved_memories: list[MemoryWithTimestamp] = dspy.InputField()
    premise_check: str = dspy.OutputField(
        desc="List the key premises embedded in the question (role titles, names, dates, events, quantities). For each, state whether the memories SUPPORT it, CONTRADICT it (memories state a different fact), or are SILENT (memories don't mention it). Only mark CONTRADICT when the memories provide a conflicting fact — silence is not contradiction. If all premises are supported or the memories are simply silent, say 'Premises verified.'"
    )
    answer: str = dspy.OutputField(
        desc="If the premise check found a CONTRADICTION (memories state a different fact than the question assumes), abstain: say 'The information provided is not enough.' and explain the discrepancy. For factual questions, answer strictly from the memories. For preference/recommendation questions, synthesize the user's stored preferences, habits, and interests into a personalized suggestion — this is expected, not fabrication."
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
        lm = dspy.LM(dspy_lm_model, api_key=os.environ.get("OPENAI_API_KEY"), cache=False)
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

        try:
            retrieval_cls = _RETRIEVAL_CLASSES[self.retrieval_type]
        except KeyError:
            raise ValueError(
                f"Unsupported retrieval_type '{self.retrieval_type}'. "
                f"Supported: {sorted(_RETRIEVAL_CLASSES)}"
            )

        memories = self.engram_client.memories.search(
            query=query,
            user_id=user_id,
            group=self.engram_group,
            retrieval_config=retrieval_cls(limit=self.retrieval_limit),
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

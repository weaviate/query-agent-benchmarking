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

from engram import (
    AsyncEngramClient,
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

class AnswerUserQueryWithMemory(dspy.Signature):
    """Answer the user's question using the retrieved memories as your knowledge base.

    Determine the question type, then apply the matching strategy:

    FACTUAL questions:
    - Answer using ONLY facts explicitly stated in the memories.
    - Only mention specific names, numbers, products, or details if they appear in the memories.
    - Do NOT fabricate or guess facts.

    PREFERENCE / RECOMMENDATION questions:
    - These ask you to USE the user's stored preferences, habits, and interests to make a personalized suggestion.
    - You SHOULD synthesize across memories to form a recommendation grounded in what the user likes, has, or has done.
    - Do NOT abstain just because no memory literally answers the question — the memories provide the ingredients for your recommendation.

    BOTH types:
    - Synthesizing facts across multiple memories is expected and encouraged.
    - BEFORE answering, identify which memories are relevant to answering the question. Then, reason through these instructions, making sure to follow them exactly, and ensure that you have done all required temporal reasoning. Finally, decide whether you have the information you need to answer the question or should abstain, again making sure to follow the instructions about this decision.
    - If you have partial information about a question, use the reasoning field to make the best possible deduction you can from the memories. You should not abstain from answering if you have partial information, as you can still provide a best-effort answer.
    - You should only abstain if the memories contain NO relevant information at all. In this case, say "The information provided is not enough." and explain why fully. Information is only not relevant if the question assumes a specific fact (role title, name, date, event, location) and the memories state a DIFFERENT fact.

    Temporal reasoning:
    - The question was asked on the date provided in `question_date`. Treat that as "today" when interpreting any time-relative terms ("now", "yesterday", "this year", etc.).
    - When memories describe the same thing at different points in time, resolve the timeline. Look for language that distinguishes past states, plans/intentions, and current states:
      * Past: "used to", "previously"
      * Plans: "plans to", "intends to", "wants to", "will"
      * Current: present tense statements of fact
    - A current-state memory supersedes a past-state or plan memory about the same topic — do NOT hedge or say "unclear."
    - If a memory describes both a past and current state, treat them as distinct facts at different points in time — do not collapse them.
    - Memories are ordered from oldest to newest. If multiple memories appear to give different information, you should interpret this as information which has been updated over time. This is not a contradiction - the memory nearest the end is the newest, and so should be considered the current state."""

    user_question: str = dspy.InputField()
    question_date: str = dspy.InputField(
        desc="The date on which this question was asked. Treat this as today's date when interpreting any time-relative language in the question or memories."
    )
    retrieved_memories: list[str] = dspy.InputField()
    reasoning: str = dspy.OutputField(
        desc="First, identify which memories are relevant to answering the question. Then, reason through the instructions above, making sure to follow them exactly, and ensure that you have done all required temporal reasoning. Finally, decide whether you have the information you need to answer the question or should abstain, again making sure to follow the instructions about this decision above."
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
        search_topics: Optional[list[str]] = None,
    ):
        api_key = engram_api_key or os.environ["ENGRAM_API_KEY"]
        self.engram_client = EngramClient(
            api_key=api_key,
            base_url=engram_base_url,
        )
        self.async_engram_client = AsyncEngramClient(
            api_key=api_key,
            base_url=engram_base_url,
        )
        self.retrieval_limit = retrieval_limit
        self.retrieval_type = retrieval_type
        self.engram_group = engram_group
        self.user_id_prefix = user_id_prefix
        self.search_topics = search_topics

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
        question_date: Optional[str] = None,
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
            topics=self.search_topics,
        )

        sorted_memories = sorted(memories, key=lambda m: m.updated_at)
        retrieved = [m.content for m in sorted_memories]

        response = self.qa_system(
            user_question=query,
            question_date=question_date or "unknown",
            retrieved_memories=retrieved,
        )

        return EngramAskResponse(
            final_answer=response.answer,
            raw_response={
                "n_memories_retrieved": len(retrieved),
                "memories": [
                    {"memory": m.content, "time_added": str(m.updated_at)}
                    for m in sorted_memories
                ],
            },
        )

    async def run_async(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        oracle_context_id: Optional[str] = None,
        question_date: Optional[str] = None,
    ) -> EngramAskResponse:
        """
        Retrieve memories from Engram and answer the question using DSPy, both async.

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

        memories = await self.async_engram_client.memories.search(
            query=query,
            user_id=user_id,
            group=self.engram_group,
            retrieval_config=retrieval_cls(limit=self.retrieval_limit),
            topics=self.search_topics,
        )

        sorted_memories = sorted(memories, key=lambda m: m.updated_at)
        retrieved = [m.content for m in sorted_memories]

        response = await self.qa_system.acall(
            user_question=query,
            question_date=question_date or "unknown",
            retrieved_memories=retrieved,
        )

        return EngramAskResponse(
            final_answer=response.answer,
            raw_response={
                "n_memories_retrieved": len(retrieved),
                "memories": [
                    {"memory": m.content, "time_added": str(m.updated_at)}
                    for m in sorted_memories
                ],
            },
        )

    async def initialize_async(self) -> None:
        pass

    async def close_async(self) -> None:
        pass

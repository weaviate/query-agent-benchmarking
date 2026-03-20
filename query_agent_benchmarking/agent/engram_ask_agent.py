from typing import Optional

from engram import EngramClient
import openai

from query_agent_benchmarking.agent.ask_agent import AskResponse


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the user's memories. "
    "Use the provided memories as context to answer the question. "
    "If the memories don't contain enough information, say so."
)


class EngramAskAgent:
    """
    Ask agent that retrieves memories from Engram and generates answers via an LLM.

    Flow: query -> Engram search (scoped by tenant_id) -> build context -> LLM answer.

    This agent does not extend BaseAgentBuilder since it doesn't need a Weaviate connection.
    It follows the same interface (run/run_async returning AskResponse) so it plugs directly
    into run_ask_queries() / run_ask_queries_async().

    For multi-tenant datasets like LongMemEval, each query's tenant_id is used as the
    Engram user_id, optionally prefixed with engram_user_id_prefix.
    """

    def __init__(
        self,
        engram_api_key: str,
        engram_base_url: str,
        engram_user_id_prefix: str = "",
        llm_model: str = "gpt-4.1",
        system_prompt: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.engram_client = EngramClient(
            api_key=engram_api_key,
            base_url=engram_base_url,
        )
        self.engram_user_id_prefix = engram_user_id_prefix
        self.llm_model = llm_model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = openai.OpenAI()

        print(f"EngramAskAgent initialized:")
        print(f"  Engram base URL: {engram_base_url}")
        print(f"  User ID prefix: {engram_user_id_prefix!r}")
        print(f"  LLM model: {llm_model}")

    def _resolve_user_id(self, tenant_id: Optional[str]) -> str:
        """Map a query's tenant_id to an Engram user_id."""
        if not tenant_id:
            raise ValueError(
                "EngramAskAgent requires tenant_id on each query to scope memory search. "
                "Ensure your dataset loader populates InMemoryAskQuery.tenant_id."
            )
        return f"{self.engram_user_id_prefix}{tenant_id}"

    def _build_context(self, memories) -> str:
        """Build a context string from Engram memory search results."""
        parts = []
        for i, memory in enumerate(memories, 1):
            parts.append(f"Memory {i}: {memory.content}")
        return "\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer from the retrieved memories using an LLM."""
        user_message = (
            f"Based on the following memories, answer the question.\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {question}"
        )
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    def run(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Run a synchronous ask query via Engram retrieval + LLM generation.

        Args:
            query: The question to answer.
            oracle_context_id: Unused, kept for interface compatibility.
            tenant_id: Maps to Engram user_id for scoping memory search.
        """
        user_id = self._resolve_user_id(tenant_id)
        memories = self.engram_client.memories.search(
            query=query,
            user_id=user_id,
        )
        context = self._build_context(memories)
        answer = self._generate_answer(query, context)

        return AskResponse(
            final_answer=answer,
            raw_response={"memories": [m.content for m in memories], "answer": answer},
        )

    async def run_async(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Run an async ask query. Currently wraps the sync implementation.

        Args:
            query: The question to answer.
            oracle_context_id: Unused, kept for interface compatibility.
            tenant_id: Maps to Engram user_id for scoping memory search.
        """
        import asyncio
        return await asyncio.to_thread(self.run, query, oracle_context_id, tenant_id)

    async def initialize_async(self):
        """No-op — Engram client is initialized in __init__."""
        pass

    async def close_async(self):
        """No-op — no persistent connections to close."""
        pass

    def close_sync(self):
        """No-op — no persistent connections to close."""
        pass

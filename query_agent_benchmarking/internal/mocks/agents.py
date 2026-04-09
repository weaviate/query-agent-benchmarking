"""Mock agent implementations for testing."""

from query_agent_benchmarking.internal.core.domain.models import ObjectID
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse


class MockSearchAgent:
    """Mock search agent that returns deterministic results."""

    def __init__(self, responses: dict[str, list[str]] | None = None):
        self.responses = responses or {}
        self.call_count = 0

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        self.call_count += 1
        ids = self.responses.get(query, [])
        return [ObjectID(object_id=id_) for id_ in ids]

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        return self.run(query, tenant)

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass


class MockAskAgent:
    """Mock ask agent that returns deterministic answers."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0

    def run(self, query: str, oracle_context_id=None, tenant_id=None) -> AskResponse:
        self.call_count += 1
        answer = self.responses.get(query, "I don't know.")
        return AskResponse(final_answer=answer)

    async def run_async(self, query: str, oracle_context_id=None, tenant_id=None) -> AskResponse:
        return self.run(query, oracle_context_id, tenant_id)

    async def initialize_async(self):
        pass

    async def close_async(self):
        pass

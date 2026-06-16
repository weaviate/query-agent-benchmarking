"""Port interface for search agents."""

from typing import Optional, Protocol, Union, runtime_checkable

from query_agent_benchmarking.internal.core.domain.models import (
    ObjectID,
    SearchAgentResponse,
)

# A search agent may return either the legacy ranked list of document IDs, or a
# richer SearchAgentResponse that also exposes the structured searches the agent
# performed. The query-execution layer normalizes both shapes.
SearchAgentResult = Union[list[ObjectID], SearchAgentResponse]


@runtime_checkable
class SearchAgent(Protocol):
    """Protocol for search agent implementations.

    Implementations must support both sync and async query execution. They may
    return either a ranked ``list[ObjectID]`` (the legacy contract) or a
    ``SearchAgentResponse`` carrying both the ranked IDs and the structured
    ``searches`` the agent issued under the hood.
    """

    def run(self, query: str, tenant: Optional[str] = None) -> SearchAgentResult:
        """Execute a synchronous search query.

        Args:
            query: The search query string.
            tenant: Optional tenant ID for multi-tenant datasets.

        Returns:
            Ranked list of retrieved document ObjectIDs, or a
            ``SearchAgentResponse`` wrapping those IDs plus per-query searches.
        """
        ...

    async def run_async(self, query: str, tenant: Optional[str] = None) -> SearchAgentResult:
        """Execute an asynchronous search query.

        Args:
            query: The search query string.
            tenant: Optional tenant ID for multi-tenant datasets.

        Returns:
            Ranked list of retrieved document ObjectIDs, or a
            ``SearchAgentResponse`` wrapping those IDs plus per-query searches.
        """
        ...

    async def initialize_async(self) -> None:
        """Initialize async resources (e.g., connections)."""
        ...

    async def close_async(self) -> None:
        """Clean up async resources."""
        ...

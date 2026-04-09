"""Port interface for search agents."""

from typing import Optional, Protocol, runtime_checkable

from query_agent_benchmarking.domain.models import ObjectID


@runtime_checkable
class SearchAgent(Protocol):
    """Protocol for search agent implementations.

    Implementations must support both sync and async query execution,
    returning ranked lists of document IDs.
    """

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        """Execute a synchronous search query.

        Args:
            query: The search query string.
            tenant: Optional tenant ID for multi-tenant datasets.

        Returns:
            Ranked list of retrieved document ObjectIDs.
        """
        ...

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        """Execute an asynchronous search query.

        Args:
            query: The search query string.
            tenant: Optional tenant ID for multi-tenant datasets.

        Returns:
            Ranked list of retrieved document ObjectIDs.
        """
        ...

    async def initialize_async(self) -> None:
        """Initialize async resources (e.g., connections)."""
        ...

    async def close_async(self) -> None:
        """Clean up async resources."""
        ...

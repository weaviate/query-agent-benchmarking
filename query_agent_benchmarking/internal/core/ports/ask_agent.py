"""Port interface for ask agents."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class AskResponse:
    """Response from an ask agent."""
    final_answer: str
    raw_response: Any = field(default=None, repr=False)


@runtime_checkable
class AskAgent(Protocol):
    """Protocol for ask agent implementations.

    Implementations must support both sync and async question answering.
    """

    def run(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        """Execute a synchronous ask query.

        Args:
            query: The question to answer.
            oracle_context_id: Optional document ID for oracle context.
            tenant_id: Optional tenant ID for multi-tenant datasets.

        Returns:
            AskResponse containing the answer.
        """
        ...

    async def run_async(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        """Execute an asynchronous ask query.

        Args:
            query: The question to answer.
            oracle_context_id: Optional document ID for oracle context.
            tenant_id: Optional tenant ID for multi-tenant datasets.

        Returns:
            AskResponse containing the answer.
        """
        ...

    async def initialize_async(self) -> None:
        """Initialize async resources (e.g., connections)."""
        ...

    async def close_async(self) -> None:
        """Clean up async resources."""
        ...

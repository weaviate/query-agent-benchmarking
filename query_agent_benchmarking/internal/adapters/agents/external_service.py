"""External HTTP service adapters implementing SearchAgent and AskAgent protocols.

These adapters enable BYOS (bring-your-own-system) evaluation by sending
HTTP POST requests to a user-provided endpoint.
"""

from typing import Optional

import httpx

from query_agent_benchmarking.internal.core.models import ObjectID
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse


class ExternalSearchService:
    """SearchAgent adapter that delegates to an external HTTP service.

    Expected request:  ``{"query": "..."}``
    Expected response: ``{"results": ["id1", "id2", ...]}``
    """

    def __init__(self, host: str):
        if not host:
            raise ValueError("host is required for ExternalSearchService")
        self.host = host

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(self.host, json={"query": query})
            response.raise_for_status()
            data = response.json()
        return [ObjectID(object_id=str(doc_id)) for doc_id in data.get("results", [])]

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(self.host, json={"query": query})
            response.raise_for_status()
            data = response.json()
        return [ObjectID(object_id=str(doc_id)) for doc_id in data.get("results", [])]

    async def initialize_async(self) -> None:
        pass

    async def close_async(self) -> None:
        pass


class ExternalAskService:
    """AskAgent adapter that delegates to an external HTTP service.

    Expected request:  ``{"question": "...", "oracle_context_id": "..."}``
    Expected response: ``{"answer": "..."}``
    """

    def __init__(self, host: str):
        if not host:
            raise ValueError("host is required for ExternalAskService")
        self.host = host

    def run(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        payload = {"question": query}
        if oracle_context_id is not None:
            payload["oracle_context_id"] = oracle_context_id

        with httpx.Client(timeout=300.0) as client:
            response = client.post(self.host, json=payload)
            response.raise_for_status()
            data = response.json()

        return AskResponse(final_answer=data.get("answer", ""), raw_response=data)

    async def run_async(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        payload = {"question": query}
        if oracle_context_id is not None:
            payload["oracle_context_id"] = oracle_context_id

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(self.host, json=payload)
            response.raise_for_status()
            data = response.json()

        return AskResponse(final_answer=data.get("answer", ""), raw_response=data)

    async def initialize_async(self) -> None:
        pass

    async def close_async(self) -> None:
        pass

from typing import Optional, Any
from dataclasses import dataclass

import httpx
from weaviate.agents.query import QueryAgent, AsyncQueryAgent

from query_agent_benchmarking.agent.base import BaseAgentBuilder
from query_agent_benchmarking.models import DocsCollection


@dataclass
class AskResponse:
    """Response from an ask query."""
    final_answer: str
    raw_response: Any  # The full response object from the agent


class AskAgentBuilder(BaseAgentBuilder):
    """
    Agent builder for ask mode operations.
    
    Supports two agent types:
    * `agent_name == "query-agent-ask"` → Wraps the Weaviate QueryAgent in Ask Mode.
    * `agent_name == "external_service"` → Sends requests to an external host for RAG evaluation.
    
    The "external_service" mode allows you to bring your own retrieval + generation system
    and use the ask infrastructure for evaluation. It sends HTTP POST requests to
    `external_service_host` with `question` and optionally `oracle_context_id`.
    """
    
    def __init__(
        self,
        agent_name: str,
        dataset_name: Optional[str] = None,
        docs_collection: Optional[DocsCollection] = None,
        agents_host: Optional[str] = None,
        use_async: bool = False,
        embedding_model: Optional[str] = None,
        external_service_host: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            use_async=use_async,
            embedding_model=embedding_model,
            system_prompt=system_prompt,
        )
        
        self.agent_name = agent_name
        self.external_service_host = external_service_host
        self.weaviate_collection = None
        
        if not use_async:
            self.initialize_sync()

    def initialize_sync(self):
        if self.agent_name == "query-agent-ask":
            self.weaviate_client = self._connect_sync()
            agent_kwargs = dict(
                client=self.weaviate_client,
                collections=[self.collection],
                agents_host=self.agents_host,
            )
            if self.system_prompt:
                agent_kwargs["system_prompt"] = self.system_prompt
            self.agent = QueryAgent(**agent_kwargs)
        elif self.agent_name == "external_service":
            # External service mode - no Weaviate connection needed
            if not self.external_service_host:
                raise ValueError("external_service_host is required for external_service mode")
            print(f"External service mode initialized with host: {self.external_service_host}")
        else:
            raise ValueError(
                f"Unknown agent_name: {self.agent_name}. "
                "Must be 'query-agent-ask' or 'external_service'"
            )

    async def initialize_async(self):
        try:
            if self.agent_name == "query-agent-ask":
                self.weaviate_client = self._connect_async()
                await self.weaviate_client.connect()
                print("Async Weaviate client connected successfully")
                agent_kwargs = dict(
                    client=self.weaviate_client,
                    collections=[self.collection],
                    agents_host=self.agents_host,
                )
                if self.system_prompt:
                    agent_kwargs["system_prompt"] = self.system_prompt
                self.agent = AsyncQueryAgent(**agent_kwargs)
                print(f"AsyncQueryAgent (ask mode) initialized for collection: {self.collection}")
                print(f"Using agents host: {self.agents_host}")
            elif self.agent_name == "external_service":
                # External service mode - no Weaviate connection needed
                if not self.external_service_host:
                    raise ValueError("external_service_host is required for external_service mode")
                print(f"External service mode initialized with host: {self.external_service_host}")
            else:
                raise ValueError(
                    f"Unknown agent_name: {self.agent_name}. "
                    "Must be 'query-agent-ask' or 'external_service'"
                )
                
        except Exception as e:
            print(f"Failed to initialize async agent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def run(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Run synchronous ask query.

        Args:
            query: The question to ask.
            oracle_context_id: Optional context ID to send to external host.
            tenant_id: Optional tenant ID for multi-tenant datasets (unused by this agent).
        """
        if self.agent_name == "query-agent-ask":
            response = self.agent.ask(query)
            return AskResponse(
                final_answer=response.final_answer,
                raw_response=response
            )
        
        elif self.agent_name == "external_service":
            # Build request payload
            payload = {"question": query}
            if oracle_context_id is not None:
                payload["oracle_context_id"] = oracle_context_id
            
            # Send request to external host
            with httpx.Client(timeout=300.0) as client:
                response = client.post(self.external_service_host, json=payload)
                response.raise_for_status()
                data = response.json()
            
            return AskResponse(
                final_answer=data.get("answer", ""),
                raw_response=data
            )

    async def run_async(
        self,
        query: str,
        oracle_context_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Run asynchronous ask query.
        
        Args:
            query: The question to ask.
            oracle_context_id: Optional context ID to send to external host.
        """
        try:
            if self.agent_name == "query-agent-ask":
                response = await self.agent.ask(query)
                return AskResponse(
                    final_answer=response.final_answer,
                    raw_response=response
                )
            
            elif self.agent_name == "external_service":
                # Build request payload
                payload = {"question": query}
                if oracle_context_id is not None:
                    payload["oracle_context_id"] = oracle_context_id
                
                # Send async request to external host
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(self.external_service_host, json=payload)
                    response.raise_for_status()
                    data = response.json()
                
                return AskResponse(
                    final_answer=data.get("answer", ""),
                    raw_response=data
                )
                
        except Exception as e:
            print(f"Ask query '{query[:50]}...' failed with error: {str(e)}")
            raise


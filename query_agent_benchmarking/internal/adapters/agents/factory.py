"""Agent factory functions.

Maps agent name strings from config to concrete adapter instances
that implement the SearchAgent or AskAgent port protocols.
"""

from typing import Optional, Sequence

from query_agent_benchmarking.internal.adapters.agents.weaviate_search import (
    WeaviateHybridSearch,
    WeaviateVectorSearch,
    WeaviateBM25Search,
    parse_agent_name,
)
from query_agent_benchmarking.internal.adapters.agents.weaviate_query_agent import (
    WeaviateQueryAgentSearch,
    WeaviateQueryAgentAsk,
)
from query_agent_benchmarking.internal.adapters.agents.external_service import (
    ExternalSearchService,
    ExternalAskService,
)


def create_search_agent(
    agent_name: str,
    *,
    dataset_name: Optional[str] = None,
    docs_collection=None,
    agents_host: Optional[str] = None,
    embedding_model: Optional[str] = None,
    text_embedding_model: Optional[str] = None,
    image_embedding_model: Optional[str] = None,
    embedding_providers: Optional[Sequence[str]] = None,
    external_service_host: Optional[str] = None,
    filtering: Optional[str] = None,
):
    """Create a SearchAgent adapter based on agent_name.

    Args:
        agent_name: One of "query-agent-search-mode", "hybrid-search",
            "vector-search", "bm25-search", or "external_service".
            Hybrid and vector search support a ``[target_vector]`` suffix,
            e.g. ``"hybrid-search[text_content_weaviate]"``.
        dataset_name: Built-in dataset name for collection resolution.
        docs_collection: Custom DocsCollection config.
        agents_host: Host URL for the Weaviate agents service.
        embedding_model: Legacy single embedding model string.
        text_embedding_model: Text embedding model.
        image_embedding_model: Image embedding model.
        embedding_providers: Explicit provider list for header injection.
        external_service_host: Host URL for external service mode.
        filtering: Query Agent search filtering strategy, "recall" (default,
            generate multiple Weaviate queries) or "precision" (generate a
            single query for the most likely interpretation). Only applies to
            "query-agent-search-mode".

    Returns:
        An object implementing the SearchAgent protocol.
    """
    base_name, target_vector_config = parse_agent_name(agent_name)

    _embedding_kwargs = dict(
        embedding_model=embedding_model,
        text_embedding_model=text_embedding_model,
        image_embedding_model=image_embedding_model,
        embedding_providers=embedding_providers,
    )

    if base_name == "query-agent-search-mode":
        return WeaviateQueryAgentSearch(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            filtering=filtering,
            **_embedding_kwargs,
        )

    if base_name == "hybrid-search":
        return WeaviateHybridSearch(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            target_vector_config=target_vector_config,
            **_embedding_kwargs,
        )

    if base_name == "vector-search":
        return WeaviateVectorSearch(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            target_vector_config=target_vector_config,
            **_embedding_kwargs,
        )

    if base_name == "bm25-search":
        return WeaviateBM25Search(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            **_embedding_kwargs,
        )

    if base_name == "external_service":
        if not external_service_host:
            raise ValueError("external_service_host is required for external_service mode")
        return ExternalSearchService(host=external_service_host)

    raise ValueError(
        f"Unknown search agent: '{base_name}'. "
        "Supported: 'query-agent-search-mode', 'hybrid-search', "
        "'vector-search', 'bm25-search', 'external_service'"
    )


def create_ask_agent(
    agent_name: str,
    *,
    dataset_name: Optional[str] = None,
    docs_collection=None,
    agents_host: Optional[str] = None,
    embedding_model: Optional[str] = None,
    text_embedding_model: Optional[str] = None,
    image_embedding_model: Optional[str] = None,
    embedding_providers: Optional[Sequence[str]] = None,
    external_service_host: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """Create an AskAgent adapter based on agent_name.

    Args:
        agent_name: One of "query-agent-ask-mode" or "external_service".
        dataset_name: Built-in dataset name for collection resolution.
        docs_collection: Custom DocsCollection config.
        agents_host: Host URL for the Weaviate agents service.
        embedding_model: Legacy single embedding model string.
        text_embedding_model: Text embedding model.
        image_embedding_model: Image embedding model.
        embedding_providers: Explicit provider list for header injection.
        external_service_host: Host URL for external service mode.
        system_prompt: System prompt for the ask agent.

    Returns:
        An object implementing the AskAgent protocol.
    """
    if agent_name == "query-agent-ask-mode":
        return WeaviateQueryAgentAsk(
            dataset_name=dataset_name,
            docs_collection=docs_collection,
            agents_host=agents_host,
            embedding_model=embedding_model,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            embedding_providers=embedding_providers,
            system_prompt=system_prompt,
        )

    if agent_name == "external_service":
        if not external_service_host:
            raise ValueError("external_service_host is required for external_service mode")
        return ExternalAskService(host=external_service_host)

    raise ValueError(
        f"Unknown ask agent: '{agent_name}'. "
        "Supported: 'query-agent-ask-mode', 'external_service'"
    )

import asyncio
import dspy

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    get_tag_values,
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse, Source
from benchmarker.src.dspy_rag.signatures import RerankResults

class SearchOnlyWithReranker(BaseRAG):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        self.reranker = dspy.Predict(RerankResults)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results with scores for reranking
        search_results, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_format="rerank"
        )
        
        print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Perform reranking
        rerank_pred = self.reranker(
            query=question,
            search_results=search_results
        )
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            # rank_id is 1-based, sources list is 0-based
            source_index = rank_id - 1
            if 0 <= source_index < len(sources):
                reranked_sources.append(sources[source_index])
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        
        # Get usage from reranker
        usage = rerank_pred.get_lm_usage() or {}
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results with scores for reranking
        search_results, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_format="rerank"
        )
        
        print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Perform reranking asynchronously
        rerank_pred = await self.reranker.acall(
            query=question,
            search_results=search_results
        )
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            source_index = rank_id - 1
            if 0 <= source_index < len(sources):
                reranked_sources.append(sources[source_index])
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        
        # Get usage from reranker
        usage = rerank_pred.get_lm_usage() or {}
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )
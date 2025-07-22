import asyncio
from typing import Optional

import dspy

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse
from benchmarker.src.dspy_rag.signatures import RerankResults

class SearchOnlyWithListwiseReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str, 
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20
    ):
        super().__init__(collection_name, target_property_name, retrieved_k=retrieved_k)
        self.reranked_k = reranked_k
        self.reranker = dspy.Predict(RerankResults)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results with scores for reranking
        search_results, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        # Perform reranking
        rerank_pred = self.reranker(
            query=question,
            search_results=search_results,
            top_k=self.reranked_k,
        )
        
        # Reorder sources based on reranking
        reranked_sources = []
        reranked_results = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            # rank_id is 1-based, sources list is 0-based
            source_index = rank_id - 1
            if 0 <= source_index < len(sources):
                reranked_sources.append(sources[source_index])
                reranked_results.append(search_results[source_index])
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        print("\nTop 5 reranked results:")
        for i, result in enumerate(reranked_results[:5], 1):
            print(f"New Rank {i} (was {result.initial_rank}).")
        
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
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        rerank_pred = await self.reranker.acall(
            query=question,
            search_results=search_results,
            top_k=self.reranked_k,
        )
        
        # Reorder sources based on reranking
        reranked_sources = []
        reranked_results = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            source_index = rank_id - 1
            if 0 <= source_index < len(sources):
                reranked_sources.append(sources[source_index])
                reranked_results.append(search_results[source_index])
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        print("\nTop 5 reranked results:")
        for i, result in enumerate(reranked_results[:5], 1):
            print(f"New Rank {i} (was {result.initial_rank}).")
        
        # Get usage from reranker
        usage = rerank_pred.get_lm_usage() or {}
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )

async def main():
    import os
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")

    test_pipeline = SearchOnlyWithListwiseReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=20,
        reranked_k=10
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    asyncio.run(main())
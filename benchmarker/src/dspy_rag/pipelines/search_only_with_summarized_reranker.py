import asyncio
from typing import Optional

import dspy

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG
from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse
from benchmarker.src.dspy_rag.signatures import SummarizeSearchRelevance, RerankWithSummaries

class SearchOnlyWithSummarizedReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str, 
        retrieved_k: Optional[int] = 5
    ):
        super().__init__(collection_name, target_property_name, retrieved_k=retrieved_k)
        self.summarizer = dspy.Predict(SummarizeSearchRelevance)
        self.reranker = dspy.Predict(RerankWithSummaries)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results
        search_results, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_format="rerank"
        )
        
        print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Summarize relevance for each result
        summaries = []
        total_usage = {}
        
        for i, (result, source) in enumerate(zip(search_results, sources)):
            summary_pred = self.summarizer(
                query=question,
                passage=result.text,
                passage_id=result.id
            )
            
            summaries.append({
                "passage_id": result.id,
                "relevance_summary": summary_pred.relevance_summary,
                "relevance_score": summary_pred.relevance_score
            })
            
            # Aggregate usage
            if hasattr(summary_pred, 'get_lm_usage'):
                usage = summary_pred.get_lm_usage() or {}
                for k, v in usage.items():
                    total_usage[k] = total_usage.get(k, 0) + v
        
        print(f"\033[96mGenerated {len(summaries)} relevance summaries\033[0m")
        
        # Perform reranking based on summaries
        rerank_pred = self.reranker(
            query=question,
            passage_summaries=summaries
        )
        
        # Aggregate reranker usage
        if hasattr(rerank_pred, 'get_lm_usage'):
            usage = rerank_pred.get_lm_usage() or {}
            for k, v in usage.items():
                total_usage[k] = total_usage.get(k, 0) + v
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            for i, source in enumerate(sources):
                if search_results[i].id == rank_id:
                    reranked_sources.append(source)
                    break
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=total_usage,
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results
        search_results, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_format="rerank"
        )
        
        print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Summarize relevance for each result in parallel
        summary_tasks = []
        for i, (result, source) in enumerate(zip(search_results, sources)):
            task = self.summarizer.acall(
                query=question,
                passage=result.text,
                passage_id=result.id
            )
            summary_tasks.append(task)
        
        # Wait for all summaries to complete
        summary_preds = await asyncio.gather(*summary_tasks)
        
        # Process summaries and aggregate usage
        summaries = []
        total_usage = {}
        
        for summary_pred in summary_preds:
            summaries.append({
                "passage_id": summary_pred.passage_id,
                "relevance_summary": summary_pred.relevance_summary,
                "relevance_score": summary_pred.relevance_score
            })
            
            # Aggregate usage
            if hasattr(summary_pred, 'get_lm_usage'):
                usage = summary_pred.get_lm_usage() or {}
                for k, v in usage.items():
                    total_usage[k] = total_usage.get(k, 0) + v
        
        print(f"\033[96mGenerated {len(summaries)} relevance summaries in parallel\033[0m")
        
        # Perform reranking based on summaries
        rerank_pred = await self.reranker.acall(
            query=question,
            passage_summaries=summaries
        )
        
        # Aggregate reranker usage
        if hasattr(rerank_pred, 'get_lm_usage'):
            usage = rerank_pred.get_lm_usage() or {}
            for k, v in usage.items():
                total_usage[k] = total_usage.get(k, 0) + v
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            for i, source in enumerate(sources):
                if search_results[i].id == rank_id:
                    reranked_sources.append(source)
                    break
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=total_usage,
        )

async def main():
    test_pipeline = SearchOnlyWithSummarizedReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=5
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
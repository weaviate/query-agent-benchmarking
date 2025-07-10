import asyncio
import dspy

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse, Source
from benchmarker.src.dspy_rag.signatures import WriteSearchQueries, RerankResults

class SearchOnlyWithQueryWriterAndRerank(BaseRAG):
    def __init__(self, collection_name: str, target_property_name: str, retrieved_k: int):
        super().__init__(collection_name, target_property_name, retrieved_k)
        self.query_writer = dspy.Predict(WriteSearchQueries)
        self.reranker = dspy.Predict(RerankResults)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        all_search_results = []
        all_sources: list[Source] = []
        for q in queries:
            search_results, sources = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_format="rerank"
            )
            all_search_results.extend(search_results)
            all_sources.extend(sources)

        print(f"\033[96mCollected {len(all_sources)} candidates from {len(queries)} queries\033[0m")

        rerank_pred = self.reranker(
            query=question,
            search_results=all_search_results
        )
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            idx = rank_id - 1
            if 0 <= idx < len(all_sources):
                reranked_sources.append(all_sources[idx])

        print(f"\033[96mAfter reranking: {len(reranked_sources)} sources\033[0m")

        usage_buckets.append(rerank_pred.get_lm_usage() or {})

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = await self.query_writer.acall(question=question)
        queries: list[str] = qw_pred.search_queries
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")
        usage_buckets = [qw_pred.get_lm_usage() or {}]

        tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_format="rerank"
            )
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        all_search_results = []
        all_sources: list[Source] = []
        for search_results, sources in results:
            all_search_results.extend(search_results)
            all_sources.extend(sources)

        print(f"\033[96mCollected {len(all_sources)} candidates from {len(queries)} queries\033[0m")

        rerank_pred = await self.reranker.acall(
            query=question,
            search_results=all_search_results
        )
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            idx = rank_id - 1
            if 0 <= idx < len(all_sources):
                reranked_sources.append(all_sources[idx])

        print(f"\033[96mAfter reranking: {len(reranked_sources)} sources\033[0m")
        usage_buckets.append(rerank_pred.get_lm_usage() or {})

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
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

    test_pipeline = SearchOnlyWithQueryWriterAndRerank(
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
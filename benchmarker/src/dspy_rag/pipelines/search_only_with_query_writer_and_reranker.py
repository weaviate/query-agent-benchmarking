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
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
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
import asyncio
import dspy

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse, Source
from benchmarker.src.dspy_rag.signatures import WriteSearchQueries

class SearchOnlyWithQueryWriter(BaseRAG):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        self.query_writer = dspy.ChainOfThought(WriteSearchQueries)
        #self.query_writer = dspy.Predict(WriteSearchQueries)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries
        #reasoning = qw_pred.reasoning
        #print(f"\033[97mReasoning:\n{reasoning}\033[0m")
        #queries.append(reasoning)
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        sources: list[Source] = []
        for q in queries:
            _, src = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
            )
            sources.extend(src)

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Generate queries asynchronously
        qw_pred = await self.query_writer.acall(question=question)
        queries: list[str] = qw_pred.search_queries
        reasoning = qw_pred.reasoning
        print(f"\033[95mReasoning:\n{reasoning}\033[0m")
        queries.append(reasoning)
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        # Execute searches concurrently
        search_tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
            )
            for q in queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        sources: list[Source] = []
        for _, src in search_results:
            sources.extend(src)

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
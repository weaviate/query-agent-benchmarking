import asyncio
import dspy

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    get_tag_values,
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse, Source
from benchmarker.src.dspy_rag.signatures import SearchQueryWithFilter, WriteSearchQueriesWithFilters

class SearchOnlyRAG(BaseRAG):
    def forward(self, question: str) -> DSPyAgentRAGResponse:
        contexts, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
        )

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        contexts, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
        )

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
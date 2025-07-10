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

class SearchOnlyWithFilteredQueryWriter(BaseRAG):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        self.tags = get_tag_values(collection_name) # -> dict[str, str]
        self.stringified_tags = "\n".join(f"{tag}: {description}" for tag, description in self.tags.items())
        self.filtered_query_writer = dspy.Predict(WriteSearchQueriesWithFilters)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        fqw_pred = self.filtered_query_writer(
            question=question, 
            filters_available=self.stringified_tags
        )
        queries: list[SearchQueryWithFilter] = fqw_pred.search_queries_with_filters
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [fqw_pred.get_lm_usage() or {}]

        sources: list[Source] = []
        for q in queries:
            _, src = weaviate_search_tool(
                query=q.search_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                tag_filter_value=q.filter
            )
            sources.extend(src)

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        search_queries = [q.search_query for q in queries]

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=search_queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Generate queries asynchronously
        fqw_pred = await self.filtered_query_writer.acall(
            question=question,
            filters_available=self.stringified_tags
        )
        queries: list[SearchQueryWithFilter] = fqw_pred.search_queries_with_filters
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")
        print("\033[92mInspecting queries...\033[0m")
        for query in queries:
            print(f"Search Query: {query.search_query}")
            print(f"Filter: {query.filter}")
            print("\033[95m" + "="*30 + "\033[0m")

        usage_buckets = [fqw_pred.get_lm_usage() or {}]

        # Execute searches concurrently
        search_tasks = [
            async_weaviate_search_tool(
                query=q.search_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                tag_filter_value=q.filter
            )
            for q in queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        sources: list[Source] = []
        for _, src in search_results:
            sources.extend(src)

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        search_queries = [q.search_query for q in queries]

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=search_queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
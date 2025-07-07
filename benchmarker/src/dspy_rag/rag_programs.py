import abc
import asyncio
from typing import Dict

import dspy

from benchmarker.src.dspy_rag.rag_signatures import (
    Source,
    SearchResultWithScore,
    DSPyAgentRAGResponse,
    WriteSearchQueries,
    WriteSearchQueriesWithFilters,
    SearchQueryWithFilter,
    RerankResults,
    GenerateAnswer
)
from benchmarker.src.dspy_rag.utils import (
    get_tag_values,
    weaviate_search_tool,
    async_weaviate_search_tool
)

class RAGAblation(dspy.Module):
    def __init__(self, collection_name: str, target_property_name: str) -> None:
        self.collection_name = collection_name
        self.target_property_name = target_property_name

    @staticmethod
    def _merge_usage(*usages: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        merged: Dict[str, Dict[str, int]] = {}
        for usage in usages:
            # Skip None values
            if usage is None:
                continue
            for lm_id, stats in usage.items():
                bucket = merged.setdefault(
                    lm_id, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["prompt_tokens"] += stats.get("prompt_tokens", 0)
                bucket["completion_tokens"] += stats.get("completion_tokens", 0)
        return merged

    @abc.abstractmethod
    def forward(self, question: str) -> DSPyAgentRAGResponse: ...
    
    @abc.abstractmethod
    async def aforward(self, question: str) -> DSPyAgentRAGResponse: ...

# --- Search Only Ablations ---

class SearchOnlyRAG(RAGAblation):
    def forward(self, question: str) -> DSPyAgentRAGResponse:
        contexts, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_dict=False,
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
            return_dict=False,
        )

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
    
class SearchOnlyWithReranker(RAGAblation):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        self.reranker = dspy.Predict(RerankResults)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        contexts, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_format="rerank"
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
            return_dict=False,
        )

        print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

class SearchOnlyWithQueryWriter(RAGAblation):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        #self.query_writer = dspy.ChainOfThought(WriteSearchQueries)
        self.query_writer = dspy.Predict(WriteSearchQueries)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        sources: list[Source] = []
        for q in queries:
            _, src = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_dict=False,
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
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        # Execute searches concurrently
        search_tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_dict=False,
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
        
class SearchOnlyWithFilteredQueryWriter(RAGAblation):
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
                return_dict=False,
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
                return_dict=False,
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

# --- End-to-End RAG Ablations ---

class VanillaRAG(RAGAblation):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        contexts, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_dict=False,
        )

        ans_pred = self.generate_answer(
            question=question,
            contexts=contexts,
        )

        usage = ans_pred.get_lm_usage() or {}
        return DSPyAgentRAGResponse(
            final_answer=ans_pred.final_answer,
            sources=sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Search asynchronously
        contexts, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_dict=False,
        )

        # Generate answer asynchronously
        ans_pred = await self.generate_answer.acall(
            question=question,
            contexts=contexts,
        )

        usage = ans_pred.get_lm_usage() or {}
        return DSPyAgentRAGResponse(
            final_answer=ans_pred.final_answer,
            sources=sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )

class SearchQueryWriter(RAGAblation):
    def __init__(self, collection_name: str, target_property_name: str):
        super().__init__(collection_name, target_property_name)
        self.query_writer = dspy.Predict(WriteSearchQueries)
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries

        # keep track of token usage for final accounting
        usage_buckets = [qw_pred.get_lm_usage() or {}]

        contexts, sources = [], []

        for q in queries:
            ctx, src = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_dict=False,
            )
            contexts.append(ctx)
            sources.extend(src)

        ans_pred = self.generate_answer(
            question=question,
            contexts="\n".join(contexts),
        )
        usage_buckets.append(ans_pred.get_lm_usage() or {})

        return DSPyAgentRAGResponse(
            final_answer=ans_pred.final_answer,
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Generate queries asynchronously
        qw_pred = await self.query_writer.acall(question=question)
        queries: list[str] = qw_pred.search_queries

        # keep track of token usage for final accounting
        usage_buckets = [qw_pred.get_lm_usage() or {}]

        # Execute all searches concurrently
        search_tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_dict=False,
            )
            for q in queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        contexts, sources = [], []
        for ctx, src in search_results:
            contexts.append(ctx)
            sources.extend(src)

        # Generate answer asynchronously
        ans_pred = await self.generate_answer.acall(
            question=question,
            contexts="\n".join(contexts),
        )
        usage_buckets.append(ans_pred.get_lm_usage() or {})

        return DSPyAgentRAGResponse(
            final_answer=ans_pred.final_answer,
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

RAG_VARIANTS = {
    "search-only":        SearchOnlyRAG,
    "search-only-with-qw":     SearchOnlyWithQueryWriter,
    "search-only-with-fqw": SearchOnlyWithFilteredQueryWriter,
    "vanilla-rag":            VanillaRAG,
    "query-writer-rag":       SearchQueryWriter
}

async def async_main():
    import os
    """Test all RAG pipeline implementations asynchronously."""
    print("\033[95mSetting up DSPy with usage tracking...\033[0m")

    lm = dspy.LM(
        'openai/gpt-4.1-mini', 
        api_key=os.getenv("OPENAI_API_KEY"),
        cache=False,
    )

    dspy.configure(lm=lm, track_usage=True)
    print(f"\033[95mDSPy configured with: {lm}\033[0m")
    
    collection_name = "FreshstackLangchain"
    target_property_name = "docs_text"
    
    # Test question
    test_question = "What is LangChain?"
    print(f"\033[96mQuestion: {test_question}\033[0m")
    
    # Test SearchOnlyRAG
    print("\n\033[95m=== Testing SearchOnlyRAG (Async) ===\033[0m")
    search_only_rag = SearchOnlyRAG(collection_name, target_property_name)
    search_only_response = await search_only_rag.acall(test_question)
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{search_only_response.sources}\033[0m")

    # Test SearchOnlyWithQueryWriter
    print("\n\033[95m=== Testing SearchOnlyWithQueryWriter (Async) ===\033[0m")
    search_only_qw_rag = SearchOnlyWithQueryWriter(collection_name, target_property_name)
    search_only_qw_response = await search_only_qw_rag.acall(test_question)
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{search_only_qw_response.sources}\033[0m")

    # Test SearchOnlyWithFilteredQueryWriter
    print("\n\033[95m=== Testing SearchOnlyWithFilteredQueryWriter (Async) ===\033[0m")
    search_only_fqw_rag = SearchOnlyWithFilteredQueryWriter(collection_name, target_property_name)
    search_only_fqw_response = await search_only_fqw_rag.acall(test_question)
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{search_only_fqw_response.sources}\033[0m")
    
    # Test VanillaRAG
    print("\n\033[95m=== Testing VanillaRAG (Async) ===\033[0m")
    vanilla_rag = VanillaRAG(collection_name, target_property_name)
    vanilla_response = await vanilla_rag.acall(test_question)
    print(f"\033[92m{vanilla_response.final_answer}\033[0m\n")
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{vanilla_response.sources}\033[0m")
    print("\033[96mUsage:\033[0m")
    print(f"\033[92m{vanilla_response.usage}\033[0m")
    
    # Test SearchQueryWriter
    print("\n\033[95m=== Testing SearchQueryWriter (Async) ===\033[0m")
    query_writer_rag = SearchQueryWriter(collection_name, target_property_name)
    query_writer_response = await query_writer_rag.acall(test_question)
    print(f"\033[92m{query_writer_response.final_answer}\033[0m\n")
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{query_writer_response.sources}\033[0m")
    print("\033[96mUsage:\033[0m")
    print(f"\033[92m{query_writer_response.usage}\033[0m")

def main():
    import os
    """Test all RAG pipeline implementations."""
    print("\033[95mSetting up DSPy with usage tracking...\033[0m")

    lm = dspy.LM(
        'openai/gpt-4.1-mini', 
        api_key=os.getenv("OPENAI_API_KEY"),
        cache=False,
    )

    dspy.configure(lm=lm, track_usage=True)
    print(f"\033[95mDSPy configured with: {lm}\033[0m")
    
    collection_name = "FreshstackLangchain"
    target_property_name = "docs_text"
    
    # Test question
    test_question = "What is LangChain?"
    print(f"\033[96mQuestion: {test_question}\033[0m")
    
    # Test SearchOnlyRAG
    print("\n\033[95m=== Testing SearchOnlyRAG ===\033[0m")
    search_only_rag = SearchOnlyRAG(collection_name, target_property_name)
    search_only_response = search_only_rag.forward(test_question)
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{search_only_response.sources}\033[0m")

    # Test SearchOnlyRAG
    print("\n\033[95m=== Testing SearchOnlyWithQueryWriter ===\033[0m")
    search_only_rag = SearchOnlyWithQueryWriter(collection_name, target_property_name)
    search_only_response = search_only_rag.forward(test_question)
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{search_only_response.sources}\033[0m")
    
    # Test VanillaRAG
    print("\n\033[95m=== Testing VanillaRAG ===\033[0m")
    vanilla_rag = VanillaRAG(collection_name, target_property_name)
    vanilla_response = vanilla_rag.forward(test_question)
    print(f"\033[92m{vanilla_response.final_answer}\033[0m\n")
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{vanilla_response.sources}\033[0m")
    print("\033[96mUsage:\033[0m")
    print(f"\033[92m{vanilla_response.usage}\033[0m")
    
    # Test SearchQueryWriter
    print("\n\033[95m=== Testing SearchQueryWriter ===\033[0m")
    query_writer_rag = SearchQueryWriter(collection_name, target_property_name)
    query_writer_response = query_writer_rag.forward(test_question)
    print(f"\033[92m{query_writer_response.final_answer}\033[0m\n")
    print("\033[96mSources:\033[0m")
    print(f"\033[92m{query_writer_response.sources}\033[0m")
    print("\033[96mUsage:\033[0m")
    print(f"\033[92m{query_writer_response.usage}\033[0m")

if __name__ == "__main__":
    # Run sync version
    # main()
    
    # Run async version
    asyncio.run(async_main())
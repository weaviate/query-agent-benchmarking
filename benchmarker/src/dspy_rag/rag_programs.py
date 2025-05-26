import abc
from typing import Dict

import dspy

from benchmarker.src.dspy_rag.rag_signatures import (
    Source,
    AgentRAGResponse,
    GenerateAnswer,
    WriteSearchQueries,
    FilterIrrelevantSearchResults,
    SummarizeSearchResults
)
from benchmarker.src.dspy_rag.dspy_rag_utils import weaviate_search_tool

class RAGAblation(abc.ABC):
    def __init__(self, collection_name: str, target_property_name: str) -> None:
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.query_writer = dspy.Predict(WriteSearchQueries)
        self.result_filter = dspy.Predict(FilterIrrelevantSearchResults)
        self.result_summarizer = dspy.Predict(SummarizeSearchResults)

        self.collection_name = collection_name
        self.target_property_name = target_property_name

    @staticmethod
    def _merge_usage(*usages: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        merged: Dict[str, Dict[str, int]] = {}
        for usage in usages:
            for lm_id, stats in usage.items():
                bucket = merged.setdefault(
                    lm_id, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["prompt_tokens"] += stats.get("prompt_tokens", 0)
                bucket["completion_tokens"] += stats.get("completion_tokens", 0)
        return merged

    @abc.abstractmethod
    def forward(self, question: str) -> AgentRAGResponse: ...

# --- Search Only Ablations ---

class SearchOnlyRAG(RAGAblation):
    def forward(self, question: str) -> AgentRAGResponse:
        _, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_dict=False,
        )

        return AgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

class SearchOnlyWithQueryWriter(RAGAblation):
    def forward(self, question: str) -> AgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries or [question]
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage()]

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

        return AgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
# --- End-to-End RAG Ablations ---

class VanillaRAG(RAGAblation):
    def forward(self, question: str) -> AgentRAGResponse:
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
        return AgentRAGResponse(
            final_answer=ans_pred.final_answer,
            sources=sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )

class SearchQueryWriter(RAGAblation):
    def forward(self, question: str) -> AgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries

        # keep track of token usage for final accounting
        usage_buckets = [qw_pred.get_lm_usage()]

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
        usage_buckets.append(ans_pred.get_lm_usage())

        return AgentRAGResponse(
            final_answer=ans_pred.final_answer,
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

def main():
    import os
    """Test all RAG pipeline implementations."""
    print("\033[95mSetting up DSPy with usage tracking...\033[0m")

    lm = dspy.LM(
        'openai/gpt-4o', 
        api_key=os.getenv("OPENAI_API_KEY"),
        cache=False,
    )

    dspy.configure(lm=lm, track_usage=True)
    print(f"\033[95mDSPy configured with: {lm}\033[0m")
    
    collection_name = "WixKB"
    target_property_name = "contents"
    
    # Test question
    test_question = "How does Wix mobile work?"
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
    main()

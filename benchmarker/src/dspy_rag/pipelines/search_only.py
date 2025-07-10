
from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse

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

async def main():
    search_rag = SearchOnlyRAG(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text"
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    search_rag_response = search_rag.forward(test_q)
    print(search_rag_response)
    search_rag_async_response = await search_rag.aforward(test_q)
    print(search_rag_async_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
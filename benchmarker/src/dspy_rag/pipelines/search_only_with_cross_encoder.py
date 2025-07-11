import asyncio
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass
import time

from sentence_transformers.cross_encoder import CrossEncoder

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse

@dataclass
class ScoredDocument:
    """Container for a document with its relevance score."""
    context: str
    source: str
    score: float
    index: int  # Original index for sorting

class SearchOnlyWithCrossEncoder(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str, 
        retrieved_k: Optional[int] = 20,
        rerank_top_n: Optional[int] = 10,
        model_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_chars_per_doc: int = 4000
    ):
        super().__init__(collection_name, target_property_name, retrieved_k=retrieved_k)
        self.rerank_top_n = rerank_top_n
        self.max_chars_per_doc = max_chars_per_doc
        
        # Initialize CrossEncoder model
        print(f"\033[94m Loading CrossEncoder model: {model_path} on {device}\033[0m")
        self.cross_encoder = CrossEncoder(model_path, device=device)
        print(f"\033[92m CrossEncoder model loaded successfully!\033[0m")

    def _filter_document(self, text: str) -> str:
        """
        Check if document exceeds limits and filter out if needed.
        
        Args:
            text: The document text
            
        Returns:
            Original text if valid, None if should be filtered out
        """
        # Handle None or empty strings
        if not text or not text.strip():
            return None
            
        text = text.strip()
        
        # If document exceeds limit, throw it out entirely
        if len(text) > self.max_chars_per_doc:
            return None
        
        return text

    def _batch_rerank(self, query: str, contexts: List[str], sources: List[str]) -> Tuple[List[str], List[str]]:
        """
        Rerank all documents in a single batch using CrossEncoder.
        
        Args:
            query: The search query
            contexts: List of document contents
            sources: List of source identifiers (UUIDs)
            
        Returns:
            Tuple of (reranked_contexts, reranked_sources)
        """
        if not contexts:
            return contexts, sources

        print(f"\033[94m Batch reranking {len(contexts)} documents...\033[0m")
        start_time = time.time()
        
        # Filter out empty or oversized documents
        valid_docs = []
        for i, (context, source) in enumerate(zip(contexts, sources)):
            filtered_context = self._filter_document(context)
            if filtered_context is not None:
                valid_docs.append({
                    'context': context,  # Keep original
                    'filtered_context': filtered_context,  # Use for scoring
                    'source': source,
                    'original_index': i
                })
            else:
                print(f"\033[93m Skipping document {i} (empty or too long: {len(context) if context else 0} chars)\033[0m")
        
        if not valid_docs:
            print(f"\033[91m No valid documents to rerank!\033[0m")
            return [], []
        
        # Prepare sentence pairs for CrossEncoder
        sentence_pairs = []
        for doc in valid_docs:
            sentence_pairs.append([query, doc['filtered_context']])
        
        # Get scores from CrossEncoder
        scores = self.cross_encoder.predict(sentence_pairs)
        
        # Create scored documents
        scored_documents = []
        for i, doc in enumerate(valid_docs):
            scored_documents.append(ScoredDocument(
                context=doc['context'],  # Original context
                source=doc['source'],
                score=float(scores[i]),
                index=doc['original_index']
            ))
        
        # Sort by score (highest first)
        scored_documents.sort(key=lambda x: x.score, reverse=True)
        
        # Take top N
        top_scored = scored_documents[:self.rerank_top_n]
        
        elapsed_time = time.time() - start_time
        print(f"\033[92m Reranked {len(valid_docs)} documents in {elapsed_time:.2f}s\033[0m")
        if top_scored:
            print(f"\033[92m Score range: {top_scored[0].score:.4f} to {top_scored[-1].score:.4f}\033[0m")
        
        # Extract contexts and sources
        reranked_contexts = [doc.context for doc in top_scored]
        reranked_sources = [doc.source for doc in top_scored]
        
        return reranked_contexts, reranked_sources

    async def _batch_rerank_async(self, query: str, contexts: List[str], sources: List[str]) -> Tuple[List[str], List[str]]:
        """
        Async wrapper for batch reranking.
        """
        # Run the synchronous batch reranking in an executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._batch_rerank,
            query, contexts, sources
        )

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # Get initial search results
        contexts, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        print(f"\033[94m Initial search returned {len(sources)} sources\033[0m")

        # Rerank the results using CrossEncoder
        reranked_contexts, reranked_sources = self._batch_rerank(question, contexts, sources)

        print(f"\033[96m Returning {len(reranked_sources)} reranked sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Get initial search results
        contexts, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        print(f"\033[94m Initial search returned {len(sources)} sources\033[0m")

        # Rerank the results using CrossEncoder (async)
        reranked_contexts, reranked_sources = await self._batch_rerank_async(question, contexts, sources)

        print(f"\033[96m Returning {len(reranked_sources)} reranked sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

async def main():
    test_pipeline = SearchOnlyWithCrossEncoder(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=20,      # Get initial results
        rerank_top_n=5,      # Then rerank to top 5
        model_path="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast, good model
        device="cpu"         # Change to "cuda" if you have GPU
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    
    print("Testing synchronous forward...")
    response = test_pipeline.forward(test_q)
    print(f"Final response sources: {len(response.sources)}")
    
    print("\nTesting asynchronous aforward...")
    async_response = await test_pipeline.aforward(test_q)
    print(f"Final async response sources: {len(async_response.sources)}")

if __name__ == "__main__":
    asyncio.run(main())
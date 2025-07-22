#!/usr/bin/env python3
import asyncio
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from benchmarker.src.dspy_rag.tools.weaviate_search_tool import (
    weaviate_search_tool,
    async_weaviate_search_tool,
)
from benchmarker.src.dspy_rag.pipelines.base_rag import BaseRAG
from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse


@dataclass
class ScoredDocument:
    context: str
    source: str
    score: float
    index: int


class SearchOnlyWithCrossEncoder(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        retrieved_k: Optional[int] = 50,
        rerank_top_n: Optional[int] = 20,
        model_path: str = "BAAI/bge-reranker-v2-m3",
        max_chars_per_doc: int = 4000,
    ):
        super().__init__(collection_name, target_property_name, retrieved_k=retrieved_k)
        self.rerank_top_n = rerank_top_n
        self.max_chars_per_doc = max_chars_per_doc

        # thread‐safety & offloading
        self.lock = Lock()
        self.executor = ThreadPoolExecutor()

        # Load cross-encoder model using sentence-transformers
        self.model = CrossEncoder(model_path)
        print(f"Loaded CrossEncoder model from {model_path}")

    def _filter_document(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        text = text.strip()
        return text if len(text) <= self.max_chars_per_doc else None

    def _extract_document_content(self, formatted: str, prop: str) -> str:
        if not formatted:
            return ""
        prefix = f"  {prop}: "
        lines = formatted.splitlines()
        out, in_block = [], False
        for line in lines:
            if line.startswith(prefix):
                out.append(line[len(prefix):])
                in_block = True
            elif in_block:
                cont = f"  {prop.split('_')[0]}"
                if line.startswith("  ") and not line.startswith(cont):
                    out.append(line[2:])
                else:
                    break
        return "\n".join(out).strip()

    def _batch_rerank(self, query: str, contexts: List[str], sources: List[str]) -> Tuple[List[str], List[str]]:
        """Rerank using sentence-transformers CrossEncoder."""
        if not contexts:
            return [], []

        start = time.time()
        valid = []
        for i, (ctx, src) in enumerate(zip(contexts, sources)):
            clean = self._extract_document_content(ctx, self.target_property_name)
            filt = self._filter_document(clean)
            if filt is not None:
                valid.append((i, ctx, filt, src))

        if not valid:
            return [], []

        # build parallel lists
        idxs, raws, docs, srcs = zip(*valid)
        
        # Create query-document pairs for the cross-encoder
        query_doc_pairs = [(query, doc) for doc in docs]

        # Get scores using sentence-transformers (much simpler!)
        scores = self.model.predict(query_doc_pairs)

        # collect & sort
        scored = [
            ScoredDocument(context=raws[i], source=srcs[i], score=scores[i], index=idxs[i])
            for i in range(len(scores))
        ]
        scored.sort(key=lambda d: d.score, reverse=True)
        top = scored[: self.rerank_top_n]

        elapsed = time.time() - start
        print(f"Reranked {len(scores)} docs in {elapsed:.2f}s; "
              f"score range {top[0].score:.4f}–{top[-1].score:.4f}")

        return [d.context for d in top], [d.source for d in top]

    def _safe_rerank(self, query, contexts, sources):
        with self.lock:
            return self._batch_rerank(query, contexts, sources)

    async def forward(self, question: str) -> DSPyAgentRAGResponse:
        ctx_dict, srcs = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="dict",
        )
        contexts = list(ctx_dict.values()) if ctx_dict else []

        # offload the entire rerank to a thread
        reranked_contexts, reranked_sources = await asyncio.wrap_future(
            self.executor.submit(self._safe_rerank, question, contexts, srcs)
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        ctx_dict, srcs = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="dict",
        )
        contexts = list(ctx_dict.values()) if ctx_dict else []

        reranked_contexts, reranked_sources = await asyncio.wrap_future(
            self.executor.submit(self._safe_rerank, question, contexts, srcs)
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )


async def main():
    pipe = SearchOnlyWithCrossEncoder(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        rerank_top_n=5,
        model_path="BAAI/bge-reranker-v2-m3",
    )

    # sanity check
    print("Sanity check:", 
          pipe.model.predict([("What is 4+4?", "8")])[0])

    # real query
    resp = await pipe.forward("How do I integrate Weaviate and Langchain?")
    print("Got sources:", resp.sources)

if __name__ == "__main__":
    asyncio.run(main())
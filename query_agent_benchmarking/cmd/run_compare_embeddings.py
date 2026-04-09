"""Thin entry point for embedding comparison — delegates to the service layer."""

from query_agent_benchmarking.internal.core.services.compare_embeddings import (
    compare_embeddings,
)

__all__ = ["compare_embeddings"]

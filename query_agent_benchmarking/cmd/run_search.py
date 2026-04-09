"""Thin entry point for search benchmarks — delegates to the service layer."""

from query_agent_benchmarking.internal.core.services.search_benchmark import (
    run_search_eval,
    run_search_evals,
)

__all__ = ["run_search_eval", "run_search_evals"]
